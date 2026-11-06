"""Basic checks for the ISP building blocks:  python test_isp.py  (or pytest)."""
import numpy as np
import torch
from PIL import Image

import classical_isp as cl
import hybrid_isp as hy
import raw_sim as rs

IMG = np.asarray(Image.open('data/kodak/kodim05.png').convert('RGB'), np.float32) / 255


def psnr(a, b):
    return 10 * np.log10(1 / np.mean((a - b) ** 2))


def clean_camera_image():
    x = torch.from_numpy(IMG).permute(2, 0, 1)[None]
    cam, _ = rs.unprocess(x, torch.ones(1, 3))
    return cam


def test_unprocess_roundtrip():
    # sRGB -> camera RGB -> back through CCM, gamma and tone curve is lossless
    back = hy.render(clean_camera_image())[0].permute(1, 2, 0).numpy()
    assert psnr(back, IMG) > 60


def test_demosaicing():
    cam = clean_camera_image()
    ref = cam[0].permute(1, 2, 0).numpy()
    raw4 = rs.pack(cam)
    bil = hy.bilinear_demosaic(raw4)[0].permute(1, 2, 0).numpy()
    mhc = cl.demosaic_mhc(rs.unpack(raw4)[0, 0].numpy())
    assert np.allclose(bil[0::2, 0::2, 0], ref[0::2, 0::2, 0])  # sampled pixels are kept
    assert psnr(mhc, ref) > psnr(bil, ref) + 3 > 28


def test_noise_estimation():
    mos, meta = rs.capture(IMG, **rs.CONDITIONS['low_light'])
    shot, read = cl.estimate_noise(cl.normalize(mos, meta))
    true_shot, true_read = meta['noise']
    ratio = np.sqrt((shot * 0.2 + read) / (true_shot * 0.2 + true_read))
    assert 0.75 < ratio < 1.33


def test_classical_pipeline_on_clean_raw():
    mos, meta = rs.capture(IMG, gain=1, shot=1e-6)
    out = cl.classical_isp(mos, meta, sharp=0)
    assert out.shape == IMG.shape and psnr(out, IMG) > 27


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
