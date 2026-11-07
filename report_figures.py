"""Extra figures for the report: extreme low light and the hybrid's intermediate states.

Uses the same captures (illuminant, noise seed) as evaluate.py.

    python report_figures.py   -> results/extreme_low_light.png, results/intermediate_states.png
"""
import matplotlib
import numpy as np
import torch
from PIL import Image

import classical_isp as cl
import hybrid_isp as hy
import raw_sim as rs
from evaluate import SCENARIOS

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def psnr(a, b):
    return 10 * np.log10(1 / np.mean((np.clip(a, 0, 1) - b) ** 2))


def capture(idx, scen):
    gt = np.asarray(Image.open(f'data/kodak/kodim{idx + 1:02d}.png').convert('RGB'), np.float32) / 255
    rng = np.random.default_rng(idx)
    wb = (float(rng.uniform(1.6, 2.6)), 1.0, float(rng.uniform(1.3, 2.1)))
    cfg = SCENARIOS[scen]
    mos, meta = rs.capture(gt, cfg['gain'], cfg['shot'], wb, seed=1000 * list(SCENARIOS).index(scen) + idx)
    return gt, mos, meta


def show(ax, img, title, cmap=None, stretch=False):
    ax.imshow(img, cmap=cmap, interpolation='nearest', **({} if stretch else {'vmin': 0, 'vmax': 1}))
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def extreme_low_light(hybrid, idx=22, y=40, x=250, s=260):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.9))
    for row, scen in enumerate(['night', 'extreme']):
        gt, mos, meta = capture(idx, scen)
        raw = np.clip(cl.normalize(mos, meta), 0, 1) ** (1 / 2.2)   # as captured: no gain
        out_c, out_h = cl.classical_isp(mos, meta), hybrid(mos, meta)
        g = SCENARIOS[scen]['gain']
        crop = lambda a: a[y:y + s, x:x + s]
        show(axes[row, 0], crop(raw), f'RAW as captured (x{g} under-exposed)', cmap='gray')
        show(axes[row, 1], crop(out_c), f'Classical ISP, {psnr(out_c, gt):.2f} dB')
        show(axes[row, 2], crop(out_h), f'Hybrid ISP, {psnr(out_h, gt):.2f} dB')
        show(axes[row, 3], crop(gt), 'Ground truth')
        axes[row, 0].text(-0.06, 0.5, f'{scen.capitalize()} (x{g})', transform=axes[row, 0].transAxes,
                          rotation=90, va='center', ha='right', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig('results/extreme_low_light.png', dpi=110)
    plt.close(fig)


def intermediate_states(core, hybrid, idx=18, y=280, x=120, s=240):
    gt, mos, meta = capture(idx, 'low_light')
    with torch.no_grad():
        xin, gain, _, _ = hy._inputs(mos, meta, balance=True)
        base = hy.bilinear_demosaic(xin[:, :4])
        resid = hy._run(core, xin)[0].abs().mean(0).numpy()
        base_srgb = hy.render(base, meta['cam2rgb'])[0].permute(1, 2, 0).numpy()
    mosaic = np.clip(cl.normalize(mos, meta) * gain, 0, 1) ** (1 / 2.2)
    sigma = np.kron(xin[0, 5].numpy(), np.ones((2, 2)))  # noise std of the Gr plane, full resolution
    out = hybrid(mos, meta)
    crop = lambda a: a[y:y + s, x:x + s]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    show(axes[0, 0], crop(mosaic), '1. Bayer mosaic after x16 gain', cmap='gray')
    show(axes[0, 1], crop(sigma), f'2. Noise map (green plane), {crop(sigma).min():.3f} to {crop(sigma).max():.3f}',
         cmap='magma', stretch=True)
    show(axes[0, 2], crop(base_srgb), f'3. Bilinear base through back end, {psnr(base_srgb, gt):.2f} dB')
    show(axes[1, 0], crop(resid), '4. Learned residual |r|, mean over RGB', cmap='magma', stretch=True)
    show(axes[1, 1], crop(out), f'5. Hybrid output, {psnr(out, gt):.2f} dB')
    show(axes[1, 2], crop(gt), '6. Ground truth')
    fig.tight_layout()
    fig.savefig('results/intermediate_states.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    core = hy.load_model('checkpoints/core.pt')
    hybrid = hy.HybridISP(core)
    extreme_low_light(hybrid)
    intermediate_states(core, hybrid)
    print('saved results/extreme_low_light.png, results/intermediate_states.png')
