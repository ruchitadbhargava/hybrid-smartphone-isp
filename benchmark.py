"""Efficiency of the learned ISP components on CPU.

Measures parameters, serialized model size, feature-map memory and latency
for a 1 MP frame. Also creates the INT8 models by post-training static
quantisation (calibrated on training crops only) and saves the deployable
INT8 Hybrid-Lite as TorchScript (checkpoints/lite_int8.pt).

    python benchmark.py     -> results/efficiency.{json,md}
"""
import glob
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import classical_isp as cl
import hybrid_isp as hy
import raw_sim as rs
from train import random_crops, synth

H = W = 1024  # 1 MP output frame


def calibration_inputs(n=64):
    files = sorted(glob.glob('data/div2k/*.png'))[:-5]
    imgs = [np.asarray(Image.open(f).convert('RGB')) for f in files[::2]]
    torch.manual_seed(7)
    return synth(random_crops(imgs, n, 128, np.random.default_rng(7)), 'core')


def size_mb(model):
    buf = io.BytesIO()
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, buf)
    else:
        torch.save(model.state_dict(), buf)
    return buf.tell() / 2 ** 20


def feature_memory_mb(model, x):
    """Total size of the feature maps produced by the layers for one frame.

    ReLUs are skipped (in-place in FP32, fused into the conv in INT8).
    """
    total = 0

    def hook(_, __, out):
        nonlocal total
        total += out.numel() * out.element_size()

    layers = [m for m in model.modules() if not list(m.children()) and not isinstance(m, nn.ReLU)]
    hooks = [m.register_forward_hook(hook) for m in layers]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total / 2 ** 20


def latency_ms(fn, *args, runs=10):
    with torch.no_grad():
        for _ in range(2):
            fn(*args)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def export_int8(calib=None):
    """Quantises Hybrid-Lite to INT8 and saves it as TorchScript (deployable model)."""
    torch.backends.quantized.engine = 'onednn'
    calib = calibration_inputs() if calib is None else calib
    lite_q = hy.quantize_int8(hy.load_model('checkpoints/lite.pt'), calib)
    torch.jit.save(torch.jit.trace(lite_q, calib[:1]), 'checkpoints/lite_int8.pt')
    return lite_q


def main():
    os.makedirs('results', exist_ok=True)
    calib = calibration_inputs()
    lite_q = export_int8(calib)
    core, e2e, lite = (hy.load_model(f'checkpoints/{n}.pt') for n in ('core', 'e2e', 'lite'))
    core_q = hy.quantize_int8(core, calib)

    x = torch.rand(1, 8, H // 2, W // 2)
    rows = []
    for name, net, fp32 in [('Learned e2e (FP32)', e2e, e2e), ('Hybrid core (FP32)', core, core),
                            ('Hybrid core (INT8)', core_q, core), ('Hybrid-Lite, distilled (FP32)', lite, lite),
                            ('Hybrid-Lite, distilled (INT8)', lite_q, lite)]:
        rows.append(dict(model=name, params_M=sum(p.numel() for p in fp32.parameters()) / 1e6,
                         size_MB=size_mb(net), feature_mem_MB=feature_memory_mb(net, x),
                         latency_ms=latency_ms(net, x)))
        print(rows[-1], flush=True)

    # complete pipelines on a 1 MP RAW frame (includes the classical stages)
    gt = np.asarray(Image.open('data/kodak/kodim01.png').convert('RGB').resize((W, H)), np.float32) / 255
    mosaic, meta = rs.capture(gt, **rs.CONDITIONS['low_light'])
    deploy = torch.jit.load('checkpoints/lite_int8.pt')
    pipes = [('Classical ISP', cl.classical_isp), ('Learned ISP (e2e)', hy.LearnedISP(e2e)),
             ('Hybrid ISP', hy.HybridISP(core)), ('Hybrid-Lite ISP (FP32)', hy.HybridISP(lite)),
             ('Hybrid-Lite ISP (INT8)', hy.HybridISP(deploy))]
    full = [dict(pipeline=n, latency_ms=latency_ms(p, mosaic, meta, runs=5)) for n, p in pipes]
    print(full, flush=True)

    res = dict(threads=torch.get_num_threads(), frame=f'{W}x{H}', models=rows, pipelines=full)
    with open('results/efficiency.json', 'w') as f:
        json.dump(res, f, indent=1)
    lines = [f'Measured on CPU ({torch.get_num_threads()} threads), {W}x{H} output frame.', '',
             '| Model | Params (M) | Size (MB) | Feature maps / frame (MB) | Latency (ms) |',
             '|---|---:|---:|---:|---:|']
    lines += [f"| {r['model']} | {r['params_M']:.3f} | {r['size_MB']:.2f} | {r['feature_mem_MB']:.0f} | "
              f"{r['latency_ms']:.0f} |" for r in rows]
    lines += ['', '| Full pipeline (RAW in, sRGB out) | Latency (ms) |', '|---|---:|']
    lines += [f"| {p['pipeline']} | {p['latency_ms']:.0f} |" for p in full]
    with open('results/efficiency.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
