"""Quality comparison of classical, learned (end-to-end) and hybrid ISP.

RAW captures of the Kodak24 images are simulated under several lighting and
colour scenarios. Every pipeline gets exactly the same RAW + metadata.

    python evaluate.py      -> results/metrics.{json,md}, results/*.png
"""
import glob
import json
import os
import time

import matplotlib
import numpy as np
import torch
from PIL import Image

import classical_isp as cl
import hybrid_isp as hy
import raw_sim as rs
from metrics import quality

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SCENARIOS = {
    **{k: dict(v) for k, v in rs.CONDITIONS.items()},
    # illuminant outside the training range (strong blue/purple cast)
    'unseen_illuminant': dict(gain=4, shot=5e-4, wb=(1.3, 1.0, 2.6)),
    # no as-shot WB and no noise profile: AWB + blind noise estimation
    'no_metadata': dict(gain=4, shot=5e-4, blind=True),
}
# (image, scenario) -> crop (y, x, size) for the visual comparisons
FIGURES = {('kodim19', 'low_light'): (60, 180, 200),
           ('kodim23', 'night'): (60, 260, 240),
           ('kodim08', 'indoor'): (40, 300, 200),
           ('kodim03', 'unseen_illuminant'): (120, 200, 260)}
ABLATION = 'Hybrid (fixed noise map)'


def pipelines():
    core, e2e, lite = (hy.load_model(f'checkpoints/{n}.pt') for n in ('core', 'e2e', 'lite'))
    torch.backends.quantized.engine = 'onednn'
    hybrid = hy.HybridISP(core)
    return {
        'Classical': cl.classical_isp,
        'Learned (e2e)': hy.LearnedISP(e2e),
        'Hybrid': hybrid,
        'Hybrid-Lite': hy.HybridISP(lite),
        'Hybrid-Lite INT8': hy.HybridISP(torch.jit.load('checkpoints/lite_int8.pt')),
        # ablation: the core always gets the noise map of the 'indoor' level,
        # i.e. no adaptation to the real noise of the frame
        ABLATION: lambda mos, meta: hybrid(mos, {**meta, 'noise': (
            5e-4 * 4 / meta['gain'], rs.read_noise(5e-4) * (4 / meta['gain']) ** 2)}),
    }


def write_tables(summary):
    scen = list(summary)
    methods = list(summary[scen[0]])
    lines = []
    for key, label, fmt, best in [('psnr', 'PSNR (dB) &uarr;', '{:.2f}', max),
                                  ('ssim', 'SSIM &uarr;', '{:.3f}', max),
                                  ('delta_e', 'Colour error &Delta;E2000 &darr;', '{:.2f}', min),
                                  ('lpips', 'LPIPS &darr;', '{:.3f}', min),
                                  ('ms', 'Runtime per image (ms, CPU)', '{:.0f}', min)]:
        lines += [f'#### {label}', '', '| Method | ' + ' | '.join(scen) + ' |',
                  '|---|' + '---:|' * len(scen)]
        top = {s: best(v[key] for m, v in summary[s].items() if m != ABLATION) for s in scen}
        for m in methods:
            cells = []
            for s in scen:
                if m not in summary[s]:
                    cells.append('-')
                    continue
                v = fmt.format(summary[s][m][key])
                cells.append(f'**{v}**' if summary[s][m][key] == top[s] and m != ABLATION else v)
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    with open('results/metrics.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def plot_robustness(summary):
    conds = [c for c in rs.CONDITIONS if c in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for m in summary[conds[0]]:
        for ax, key in zip(axes, ('psnr', 'delta_e', 'lpips')):
            ax.plot(conds, [summary[c][m][key] for c in conds], 'o-', label=m,
                    ls='--' if m == ABLATION else '-')
    for ax, title in zip(axes, ('PSNR (dB), higher is better', 'Delta E 2000, lower is better',
                                'LPIPS, lower is better')):
        ax.set_title(title)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([f"{c}\n(x{rs.CONDITIONS[c]['gain']})" for c in conds])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Robustness to illumination / noise (digital gain in brackets; x64 is outside the training range)')
    fig.tight_layout()
    fig.savefig('results/robustness.png', dpi=120)
    plt.close(fig)


def plot_examples(keep):
    for (name, scen), (y, x, s) in FIGURES.items():
        items = [(m, *keep[(name, scen, m)]) for m in ['Ground truth', 'Classical', 'Learned (e2e)', 'Hybrid',
                                                       'Hybrid-Lite INT8'] if (name, scen, m) in keep]
        fig, axes = plt.subplots(1, len(items), figsize=(3 * len(items), 3.4))
        for ax, (m, img, q) in zip(axes, items):
            ax.imshow(np.clip(img[y:y + s, x:x + s], 0, 1))
            ax.set_title(m if q is None else f"{m}\n{q['psnr']:.2f} dB, dE {q['delta_e']:.2f}", fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{name} - {scen}')
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(f'results/example_{name}_{scen}.png', dpi=110)
        plt.close(fig)


def main():
    os.makedirs('results', exist_ok=True)
    files = sorted(glob.glob('data/kodak/*.png'))
    methods = pipelines()
    scores = {s: {m: [] for m in methods} for s in SCENARIOS}
    keep = {}
    for k, (scen, cfg) in enumerate(SCENARIOS.items()):
        for i, f in enumerate(files):
            name = os.path.basename(f)[:-4]
            gt = np.asarray(Image.open(f).convert('RGB'), np.float32) / 255
            rng = np.random.default_rng(i)  # same illuminant per image in every scenario
            wb = cfg.get('wb') or (float(rng.uniform(1.6, 2.6)), 1.0, float(rng.uniform(1.3, 2.1)))
            mosaic, meta = rs.capture(gt, cfg['gain'], cfg['shot'], wb, seed=1000 * k + i)
            if cfg.get('blind'):
                meta = {**meta, 'wb': None, 'noise': None}
            for m, isp in methods.items():
                if m == ABLATION and scen not in rs.CONDITIONS:
                    continue
                t = time.perf_counter()
                out = isp(mosaic, meta)
                ms = (time.perf_counter() - t) * 1000
                q = {**quality(out, gt), 'ms': ms}
                scores[scen][m].append(q)
                if (name, scen) in FIGURES:
                    keep[(name, scen, m)] = (out, q)
            if (name, scen) in FIGURES:
                keep[(name, scen, 'Ground truth')] = (gt, None)
        print(scen, {m: round(np.mean([q['psnr'] for q in v]), 2) for m, v in scores[scen].items() if v},
              flush=True)

    summary = {s: {m: {key: float(np.mean([q[key] for q in v])) for key in v[0]}
                   for m, v in d.items() if v} for s, d in scores.items()}
    with open('results/metrics.json', 'w') as f:
        json.dump(summary, f, indent=1)
    write_tables(summary)
    plot_robustness(summary)
    plot_examples(keep)


if __name__ == '__main__':
    main()
