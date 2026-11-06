"""Image quality metrics: PSNR, SSIM, colour error (CIEDE2000) and LPIPS."""
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips = None


def lpips_distance(a, b):
    global _lpips
    if _lpips is None:
        import lpips
        _lpips = lpips.LPIPS(net='alex', verbose=False).eval()
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() * 2 - 1
    with torch.no_grad():
        return float(_lpips(t(a), t(b)))


def quality(out, gt):
    """All metrics for one (H,W,3) float image in [0,1] against its ground truth."""
    out = out.astype(np.float32)
    return dict(
        psnr=float(peak_signal_noise_ratio(gt, out, data_range=1)),
        ssim=float(structural_similarity(gt, out, channel_axis=2, data_range=1)),
        delta_e=float(deltaE_ciede2000(rgb2lab(gt), rgb2lab(out)).mean()),
        lpips=lpips_distance(out, gt),
    )


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    gt = rng.random((64, 64, 3)).astype(np.float32)
    same = quality(gt, gt)
    assert same['psnr'] > 99 and same['delta_e'] < 1e-3 and same['lpips'] < 1e-3
    noisy = quality(np.clip(gt + 0.05 * rng.standard_normal(gt.shape), 0, 1).astype(np.float32), gt)
    assert 20 < noisy['psnr'] < 30 and noisy['ssim'] < same['ssim']
    print('ok', noisy)
