"""Learned and hybrid ISP pipelines.

Hybrid ISP (the proposed design):
    classical front end : BLC, normalisation, exposure gain, white balance,
                          noise profile (metadata or blind estimate)
    learned core        : U-Net does joint demosaicing + denoising on packed
                          Bayer, conditioned on a noise map, as a residual over
                          bilinear demosaicing
    classical back end  : camera CCM, gamma, tone curve, noise-aware sharpening

Learned ISP (end-to-end baseline): the same U-Net maps RAW (+ noise map)
straight to sRGB and has to learn WB, colour correction and tone as well.

Usage:  python hybrid_isp.py photo.dng -o out.png
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

import classical_isp as cl
import raw_sim as rs
from model import UNet


# ---------------------------------------------------------------- shared tensor ops

def net_input(raw4, gain, wb, shot, read):
    """Exposure-normalised (and white-balanced) packed Bayer + per-plane noise std map.

    raw4: (N,4,h,w) normalised RAW planes, gain/shot/read: (N,), wb: (N,3).
    Noise model after gain g and WB gain w:  var = w*g*shot*x + (w*g)^2 * read
    """
    g = gain.view(-1, 1, 1, 1)
    w = wb[:, [0, 1, 1, 2]].view(-1, 4, 1, 1)
    x = (raw4 * g * w).clamp(max=1)
    var = shot.view(-1, 1, 1, 1) * g * w * x.clamp(min=0) + read.view(-1, 1, 1, 1) * (g * w) ** 2
    return torch.cat([x, var.sqrt()], 1)


_K_RB = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
_K_G = torch.tensor([[0., 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
_SEL = torch.tensor([[1., 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])  # RGGB plane -> colour


def bilinear_demosaic(raw4):
    """Packed RGGB (N,4,h,w) -> RGB (N,3,2h,2w), normalised convolution (exact at borders)."""
    _, _, h, w = raw4.shape
    mask = F.pixel_shuffle(_SEL.to(raw4).view(1, 12, 1, 1).expand(1, 12, h, w), 2)
    k = torch.stack([_K_RB, _K_G, _K_RB])[:, None].to(raw4)
    sparse = F.pixel_shuffle(raw4, 2) * mask
    return F.conv2d(sparse, k, padding=1, groups=3) / F.conv2d(mask, k, padding=1, groups=3)


def render(cam, ccm=rs.CAM2RGB):
    """Classical back end (differentiable): camera RGB -> CCM -> gamma -> tone curve."""
    return rs.linear_to_srgb(rs.apply_ccm(cam, torch.as_tensor(ccm, dtype=torch.float32)).clamp(0, 1))


# ---------------------------------------------------------------- models

def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = UNet(ck['width'])
    net.load_state_dict(ck['state'])
    return net.eval()


def quantize_int8(net, calib):
    """Post-training static INT8 quantisation (FX graph mode, oneDNN backend).

    The first conv (raw input range) and the output head (fine residual
    values) are kept in FP32, everything else runs in INT8.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = 'onednn'
    qmap = get_default_qconfig_mapping('onednn').set_module_name('e1.0', None).set_module_name('head', None)
    prepared = prepare_fx(copy.deepcopy(net).eval(), qmap, (calib[:1],))
    with torch.no_grad():
        for x in calib.split(4):
            prepared(x)
    return convert_fx(prepared)


def _device(net):
    p = next(iter(net.parameters()), None)
    return p.device if p is not None else torch.device('cpu')


def _run(net, x):
    """Pads the packed input to a multiple of 8 (three poolings) and crops back."""
    h, w = x.shape[-2:]
    xp = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='reflect')
    return net(xp)[..., :2 * h, :2 * w]


def _inputs(mosaic, meta, balance):
    m = cl.normalize(mosaic, meta)
    gain, wb, noise = cl.resolve_meta(m, meta if balance else {**meta, 'wb': (1.0, 1.0, 1.0)})
    raw4 = F.pixel_unshuffle(torch.from_numpy(m)[None, None], 2)
    t = lambda v: torch.tensor([v], dtype=torch.float32)
    x = net_input(raw4, t(gain), torch.tensor([wb], dtype=torch.float32), t(noise[0]), t(noise[1]))
    return x, gain, wb, noise


class HybridISP:
    def __init__(self, net, sharp=0.15):
        self.net, self.sharp = net, sharp

    @torch.no_grad()
    def __call__(self, mosaic, meta):
        x, gain, wb, noise = _inputs(mosaic, meta, balance=True)
        x = x.to(_device(self.net))
        cam = bilinear_demosaic(x[:, :4]) + _run(self.net, x)
        out = render(cam, meta['cam2rgb'])[0].permute(1, 2, 0).cpu().numpy()
        sigma = cl.display_sigma(gain, wb, noise, meta['cam2rgb'])
        return np.clip(cl.sharpen(out, self.sharp / (1 + 30 * sigma)), 0, 1)


class LearnedISP:
    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def __call__(self, mosaic, meta):
        x, *_ = _inputs(mosaic, meta, balance=False)
        return _run(self.net, x.to(_device(self.net)))[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# ---------------------------------------------------------------- real RAW files

def load_raw(path):
    """Reads a camera RAW / DNG with rawpy -> (RGGB mosaic, metadata).

    Black/white levels are applied here (per CFA channel), the CFA is shifted
    to RGGB, and the CCM is built from the camera's XYZ->camera matrix.
    Exposure gain and noise profile are left to be estimated.
    """
    import rawpy
    with rawpy.imread(path) as r:
        mos = r.raw_image_visible.astype(np.float32)
        pat = r.raw_pattern
        black = np.array(r.black_level_per_channel, np.float32)
        idx = np.tile(pat, (mos.shape[0] // 2 + 1, mos.shape[1] // 2 + 1))[:mos.shape[0], :mos.shape[1]]
        mos = (mos - black[idx]) / (r.white_level - black[idx])
        colors = np.array([r.color_desc.decode()[i] for i in pat.ravel()]).reshape(2, 2)
        dy, dx = (int(v) for v in np.argwhere(colors == 'R')[0])
        wb = np.array(r.camera_whitebalance[:3], np.float32)
        xyz2cam = np.array(r.rgb_xyz_matrix[:3], np.float32)
        if xyz2cam.any():
            rgb2cam = xyz2cam @ rs.RGB2XYZ.numpy()
            cam2rgb = np.linalg.inv(rgb2cam / rgb2cam.sum(1, keepdims=True))
        else:  # some DNGs only give LibRaw's ready-made camera->sRGB matrix
            cam2rgb = np.array(r.color_matrix, np.float32)[:, :3]
    mos = mos[dy:, dx:]
    mos = mos[:mos.shape[0] // 2 * 2, :mos.shape[1] // 2 * 2]
    meta = dict(black=0.0, white=1.0, gain=None, noise=None, cam2rgb=cam2rgb,
                wb=tuple(float(v) for v in wb / wb[1]) if wb.all() else None)
    return mos, meta


if __name__ == '__main__':
    from PIL import Image
    ap = argparse.ArgumentParser(description='Process a RAW/DNG file with the hybrid ISP')
    ap.add_argument('raw')
    ap.add_argument('-o', '--out', default='out.png')
    ap.add_argument('--model', default='checkpoints/core.pt')
    ap.add_argument('--classical', action='store_true', help='use the classical pipeline instead')
    args = ap.parse_args()
    mos, meta = load_raw(args.raw)
    out = cl.classical_isp(mos, meta) if args.classical else HybridISP(load_model(args.model))(mos, meta)
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(args.out)
    print('saved', args.out)
