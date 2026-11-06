"""Training of the learned ISP components on synthetic RAW from DIV2K.

  core : hybrid ISP core (demosaic + denoise -> camera RGB), from scratch.
         The loss is taken after the classical back end (CCM, gamma, tone),
         so the network is optimised for the final sRGB output.
  e2e  : end-to-end learned ISP (RAW -> sRGB). Adapted / fine-tuned from the
         core weights, so it only has to learn WB, colour and tone on top.
  lite : small core trained by knowledge distillation from the core.

  python train.py core --width 24 --steps 4000
  python train.py e2e  --width 24 --steps 2500 --init checkpoints/core.pt
  python train.py lite --width 12 --steps 3000 --teacher checkpoints/core.pt
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import raw_sim as rs
from hybrid_isp import bilinear_demosaic, load_model, net_input, render
from model import UNet


def random_crops(imgs, n, p, rng):
    out = []
    for _ in range(n):
        im = imgs[rng.integers(len(imgs))]
        y, x = rng.integers(im.shape[0] - p + 1), rng.integers(im.shape[1] - p + 1)
        c = im[y:y + p, x:x + p]
        if rng.random() < 0.5:
            c = c[:, ::-1]
        if rng.random() < 0.5:
            c = c[::-1]
        if rng.random() < 0.5:
            c = c.transpose(1, 0, 2)
        out.append(np.ascontiguousarray(c))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255


def synth(srgb, task, wb_jitter=0.05):
    """Clean sRGB batch -> network input from a random simulated capture."""
    n = srgb.shape[0]
    wb, gain, shot, read = rs.sample_conditions(n)
    _, raw4 = rs.unprocess(srgb, wb)
    v = lambda t: t.view(-1, 1, 1, 1)
    dn = rs.add_sensor_noise(raw4, v(gain), v(shot), v(read))
    raw4 = (dn - rs.BLACK) / (rs.WHITE - rs.BLACK)
    if task == 'e2e':
        wb_in = torch.ones_like(wb)
    else:  # AWB is never perfect: small gain errors teach the core to tolerate them
        jit = torch.exp(wb_jitter * torch.randn(n, 3))
        jit[:, 1] = 1
        wb_in = wb * jit
    return net_input(raw4, gain, wb_in, shot, read)


def forward(net, x, task):
    if task == 'e2e':
        return net(x)
    return render(bilinear_demosaic(x[:, :4]) + net(x))


@torch.no_grad()
def validate(net, x, y, task):
    out = forward(net.eval(), x, task).clamp(0, 1)
    net.train()
    mse = ((out - y) ** 2).mean((1, 2, 3))
    return float((10 * torch.log10(1 / mse)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['core', 'e2e', 'lite'])
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--init', help='checkpoint to fine-tune from')
    ap.add_argument('--teacher', help='teacher checkpoint (knowledge distillation)')
    ap.add_argument('--name', help='output checkpoint name (default: kind)')
    args = ap.parse_args()
    task = 'e2e' if args.kind == 'e2e' else 'core'
    name = args.name or args.kind
    os.makedirs('checkpoints', exist_ok=True)

    files = sorted(glob.glob('data/div2k/*.png'))
    imgs = [np.asarray(Image.open(f).convert('RGB')) for f in files]
    train_imgs, val_imgs = imgs[:-5], imgs[-5:]   # last 5 images held out

    torch.manual_seed(123)
    val_y = random_crops(val_imgs, 20, 256, np.random.default_rng(123))
    val_x = synth(val_y, task)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    net = UNet(args.width)
    if args.init:
        net.load_state_dict(load_model(args.init).state_dict())
    teacher = load_model(args.teacher) if args.teacher else None
    opt = torch.optim.Adam(net.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 100)

    log, t0 = [], time.time()
    print(f'{name}: {sum(p.numel() for p in net.parameters()) / 1e6:.3f}M params, '
          f'val psnr at start {validate(net, val_x, val_y, task):.2f}', flush=True)
    for step in range(1, args.steps + 1):
        y = random_crops(train_imgs, args.batch, args.patch, rng)
        x = synth(y, task)
        out = forward(net, x, task)
        loss = F.l1_loss(out, y)
        if teacher is not None:
            with torch.no_grad():
                soft = forward(teacher, x, task)
            loss = 0.5 * loss + 0.5 * F.l1_loss(out, soft)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            log.append(dict(step=step, loss=round(loss.item(), 5),
                            val_psnr=round(validate(net, val_x, val_y, task), 3),
                            minutes=round((time.time() - t0) / 60, 2)))
            print(json.dumps(log[-1]), flush=True)
            torch.save(dict(width=args.width, kind=task, state=net.state_dict(), log=log),
                       f'checkpoints/{name}.pt')


if __name__ == '__main__':
    main()
