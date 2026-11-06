"""RAW sensor simulation and the rendering ops shared by all pipelines.

sRGB images are "unprocessed" into camera RAW following Brooks et al.,
"Unprocessing Images for Learned Raw Denoising" (CVPR 2019):
    sRGB -> inverse tone curve -> inverse gamma -> sRGB->camera CCM
         -> inverse white balance -> RGGB mosaic -> exposure (illumination)
         -> shot/read noise -> 10-bit quantisation with black level.
Every step is invertible, so the original sRGB image is the ground truth.
"""
import numpy as np
import torch
import torch.nn.functional as F

# Sensor colour response: XYZ->camera matrix of a real camera (from the
# unprocessing paper) times sRGB(D65)->XYZ, rows normalised so white stays white.
XYZ2CAM = torch.tensor([[1.0234, -0.2969, -0.2266],
                        [-0.5625, 1.6328, -0.0469],
                        [-0.0703, 0.2188, 0.6406]])
RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
RGB2CAM = XYZ2CAM @ RGB2XYZ
RGB2CAM = RGB2CAM / RGB2CAM.sum(1, keepdim=True)
CAM2RGB = torch.linalg.inv(RGB2CAM)  # the camera's colour correction matrix

BLACK, WHITE = 64, 1023  # 10-bit sensor

# Test-time capture conditions. gain = digital gain applied to reach normal
# brightness (1/exposure), shot = sensor shot-noise level. Training covers
# gain 1..32, so "extreme" is outside the training range on purpose.
CONDITIONS = {
    'daylight': dict(gain=1, shot=2e-4),
    'indoor': dict(gain=4, shot=5e-4),
    'low_light': dict(gain=16, shot=5e-4),
    'night': dict(gain=32, shot=1e-3),
    'extreme': dict(gain=64, shot=1e-3),
}


def read_noise(shot):
    # read noise follows shot noise roughly log-linearly (fit from Brooks et al.)
    return np.exp(2.18 * np.log(shot) + 1.2)


def srgb_to_linear(x):
    x = x.clamp(0, 1)
    x = 0.5 - torch.sin(torch.asin(1 - 2 * x) / 3)  # inverse smoothstep tone curve
    return x.clamp(min=1e-4) ** 2.2


def linear_to_srgb(x):
    """Gamma correction followed by a global S-curve tone map."""
    x = x.clamp(1e-4, 1) ** (1 / 2.2)
    return 3 * x ** 2 - 2 * x ** 3


def apply_ccm(x, m):
    return torch.einsum('ij,njhw->nihw', m.to(x), x)


def pack(x):
    """(N,3,H,W) image -> (N,4,H/2,W/2) RGGB planes [R, Gr, Gb, B]."""
    return F.pixel_unshuffle(x, 2)[:, [0, 5, 6, 11]]


def unpack(x):
    """(N,4,H/2,W/2) RGGB planes -> (N,1,H,W) Bayer mosaic."""
    return F.pixel_shuffle(x, 2)


def unprocess(srgb, wb):
    """sRGB batch -> (white-balanced camera-space linear image, packed RAW before noise).

    wb: (N,3) white balance gains (r, 1, b) of the simulated illuminant.
    """
    cam = apply_ccm(srgb_to_linear(srgb), RGB2CAM)
    inv = (1 / wb)[..., None, None]
    # smoothly keep near-white pixels from losing brightness (as in Brooks et al.)
    mask = ((cam.mean(1, keepdim=True) - 0.9).clamp(min=0) / 0.1) ** 2
    inv = torch.maximum(mask + (1 - mask) * inv, inv)
    return cam, pack(cam * inv)


def add_sensor_noise(raw4, gain, shot, read, generator=None):
    """Under-exposure by 1/gain, heteroscedastic noise and 10-bit quantisation -> DN."""
    x = raw4 / gain
    noise = torch.randn(x.shape, generator=generator, device=x.device)
    x = x + noise * (shot * x.clamp(min=0) + read).sqrt()
    return (x * (WHITE - BLACK) + BLACK).round().clamp(0, WHITE)


def sample_conditions(n):
    """Random illumination / noise / illuminant per training sample."""
    gain = 2 ** (5 * torch.rand(n))  # 1x .. 32x
    log_shot = torch.empty(n).uniform_(np.log(1e-4), np.log(1e-3))
    shot = log_shot.exp()
    read = (2.18 * log_shot + 1.2 + 0.26 * torch.randn(n)).exp()
    wb = torch.stack([torch.empty(n).uniform_(1.6, 2.6), torch.ones(n),
                      torch.empty(n).uniform_(1.3, 2.1)], 1)
    return wb, gain, shot, read


def capture(srgb, gain=1.0, shot=2e-4, wb=(2.1, 1.0, 1.7), seed=0):
    """Simulates a phone capture of one sRGB image (H,W,3 float in [0,1]).

    Returns the Bayer mosaic in DN (uint16) and the metadata a real camera
    would store with it (black/white level, as-shot WB, gain, noise profile, CCM).
    """
    read = float(read_noise(shot))
    x = torch.from_numpy(np.ascontiguousarray(srgb)).permute(2, 0, 1)[None].float()
    _, raw4 = unprocess(x, torch.tensor([wb], dtype=torch.float32))
    dn = add_sensor_noise(raw4, gain, shot, read, torch.Generator().manual_seed(seed))
    mosaic = unpack(dn)[0, 0].numpy().astype(np.uint16)
    meta = dict(black=BLACK, white=WHITE, wb=tuple(wb), gain=float(gain),
                noise=(shot, read), cam2rgb=CAM2RGB.numpy())
    return mosaic, meta
