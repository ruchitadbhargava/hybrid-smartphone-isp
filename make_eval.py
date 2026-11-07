"""Writes the submission file eval_P2_KuckCoders.md from results/metrics.json and
results/efficiency.json (run evaluate.py and benchmark.py first).

First line: all team USNs, comma separated. Metrics follow the order of the
problem statement: PSNR, SSIM, Delta E, LPIPS, parameters, model size,
inference latency, memory.
"""
import json

import numpy as np

USNS = ['1RUA24CSE0378', '1RUA24CSE0457', '1RUA24CSE0346', '1RUA24CSE0379', '1RUA24CSE0394']
OUT = 'eval_P2_KuckCoders.md'
# method in metrics.json -> (label, model row in efficiency.json, pipeline row in efficiency.json)
METHODS = [('Classical', 'Classical ISP (baseline)', None, 'Classical ISP'),
           ('Learned (e2e)', 'Learned ISP, end-to-end U-Net', 'Learned e2e (FP32)', 'Learned ISP (e2e)'),
           ('Hybrid', 'Hybrid ISP (proposed)', 'Hybrid core (FP32)', 'Hybrid ISP'),
           ('Hybrid-Lite', 'Hybrid-Lite, distilled (FP32)', 'Hybrid-Lite, distilled (FP32)', 'Hybrid-Lite ISP (FP32)'),
           ('Hybrid-Lite INT8', 'Hybrid-Lite, distilled (INT8)', 'Hybrid-Lite, distilled (INT8)', 'Hybrid-Lite ISP (INT8)')]


def main():
    q = json.load(open('results/metrics.json'))
    eff = json.load(open('results/efficiency.json'))
    models = {m['model']: m for m in eff['models']}
    pipes = {p['pipeline']: p for p in eff['pipelines']}
    scen = list(q)
    mean = lambda m, k: np.mean([q[s][m][k] for s in scen])

    lines = [','.join(USNS), '',
             '# Comparative evaluation: P2, Team Kuck coders', '',
             'Intelligent and Computationally Efficient Smartphone ISP. Test set: Kodak24 converted to RAW under '
             f'{len(scen)} scenarios ({24 * len(scen)} captures). Quality = mean over all captures. Latency and memory: '
             f"{eff['frame']} frame on an Intel i5-10210U laptop CPU ({eff['threads']} threads).", '',
             '| Approach | PSNR (dB) | SSIM | Delta E 2000 | LPIPS | Parameters | Model size (MB) '
             '| Inference latency (ms, full pipeline) | Memory, feature maps per frame (MB) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, label, model, pipe in METHODS:
        mo = models.get(model)
        params = '%.2f M' % mo['params_M'] if mo else 'none (no learned weights)'
        size = '%.2f' % mo['size_MB'] if mo else '-'
        memory = '%.0f' % mo['feature_mem_MB'] if mo else '-'
        lines.append(f"| {label} | {mean(m, 'psnr'):.2f} | {mean(m, 'ssim'):.3f} | {mean(m, 'delta_e'):.2f} "
                     f"| {mean(m, 'lpips'):.3f} | {params} | {size} | {pipes[pipe]['latency_ms']:.0f} | {memory} |")

    lines += ['', '## Per scenario', '']
    for key, name, fmt in [('psnr', 'PSNR (dB)', '{:.2f}'), ('ssim', 'SSIM', '{:.3f}'),
                           ('delta_e', 'Delta E 2000', '{:.2f}'), ('lpips', 'LPIPS', '{:.3f}')]:
        lines += [f'**{name}**', '', '| Approach | ' + ' | '.join(scen) + ' |', '|---|' + '---:|' * len(scen)]
        lines += [f'| {label} | ' + ' | '.join(fmt.format(q[s][m][key]) for s in scen) + ' |'
                  for m, label, _, _ in METHODS]
        lines.append('')
    lines += ['Scenarios: daylight x1, indoor x4, low_light x16, night x32 and extreme x64 digital gain '
              '(x64 is outside the training range); unseen_illuminant uses white-balance gains outside the '
              'training range; no_metadata hides the white balance and noise profile so both are estimated.',
              '', 'Reproduce: `python evaluate.py`, `python benchmark.py`, then `python make_eval.py`.', '']
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
