"""Classical ISP stages (NumPy / OpenCV) and the conventional baseline pipeline.

All functions work on float32 images. Bayer data is always RGGB here; real
RAW files with other layouts are shifted to RGGB when loaded.
"""
import cv2
import numpy as np
from scipy.optimize import nnls

RGB2YCC = np.array([[0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]], np.float32)  # BT.601 full range
YCC2RGB = np.linalg.inv(RGB2YCC).astype(np.float32)


# ---------------------------------------------------------------- RAW preprocessing

def normalize(mosaic, meta):
    """Black level subtraction + white level normalisation (pre-gain, ~[0,1])."""
    return (mosaic.astype(np.float32) - meta['black']) / (meta['white'] - meta['black'])


def planes(m):
    """RGGB mosaic -> R, Gr, Gb, B planes (views)."""
    return m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2]


def auto_exposure(m, target=0.18):
    """Digital gain that brings mean green to mid grey (used when gain is unknown)."""
    _, gr, gb, _ = planes(m)
    return float(np.clip(target / max((gr.mean() + gb.mean()) / 2, 1e-4), 1, 64))


def gray_world(m, p=4):
    """Shades-of-grey AWB on the RAW planes (p=1 grey world, large p -> white patch)."""
    r, gr, gb, b = planes(m)
    ok = np.maximum(np.maximum(r, gr), np.maximum(gb, b)) < 0.95  # skip clipped pixels
    norm = lambda c: (np.clip(c[ok], 0, None) ** p).mean() ** (1 / p)
    g = (norm(gr) + norm(gb)) / 2
    return float(g / norm(r)), 1.0, float(g / norm(b))


_HP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6  # Immerkaer, unit noise gain


def estimate_noise(m, block=8, bins=10):
    """Blind noise profile estimate: fits var(x) = shot * x + read.

    Each Bayer plane is high-pass filtered (cancels smooth signal) and split
    into 8x8 blocks. Per brightness bin, the flattest 5% of blocks give the
    noise variance; texture only adds variance so it is mostly ignored.
    1.77 corrects the low-percentile bias (calibrated on flat test charts).
    """
    xs, vs = [], []
    for p in planes(m):
        p = np.ascontiguousarray(p)
        r = cv2.filter2D(p, -1, _HP)[1:-1, 1:-1]
        h, w = (r.shape[0] // block) * block, (r.shape[1] // block) * block
        blocks = lambda a: a[:h, :w].reshape(h // block, block, w // block, block).mean((1, 3)).ravel()
        vs.append(blocks(r * r))
        xs.append(blocks(p[1:-1, 1:-1]))
    x, v = np.concatenate(xs), np.concatenate(vs)
    ok = (x > -0.02) & (x < 0.9)  # black offset keeps negative noise, so dark blocks are usable
    x, v = x[ok], v[ok]
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    X, V = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x <= hi)
        if s.sum() > 20:
            X.append(max(np.median(x[s]), 0))
            V.append(np.percentile(v[s], 5) * 1.77)
    shot, read = nnls(np.stack([X, np.ones(len(X))], 1), np.array(V))[0]
    return max(float(shot), 1e-6), max(float(read), 1e-9)


def resolve_meta(m, meta):
    """Uses camera metadata when present, otherwise estimates it from the image.

    This is what makes the pipelines adapt to unknown exposure, illuminant and
    sensor noise (e.g. RAW files without a noise profile).
    """
    gain = meta.get('gain') or auto_exposure(m)
    wb = meta.get('wb') or gray_world(m)
    noise = meta.get('noise') or estimate_noise(m)
    return gain, wb, noise


def apply_wb(m, wb):
    m = m.copy()
    m[0::2, 0::2] *= wb[0]
    m[1::2, 1::2] *= wb[2]
    return m


# ---------------------------------------------------------------- demosaicing

# Malvar-He-Cutler (2004) gradient-corrected bilinear kernels
_G_AT_RB = np.array([[0, 0, -1, 0, 0],
                     [0, 0, 2, 0, 0],
                     [-1, 2, 4, 2, -1],
                     [0, 0, 2, 0, 0],
                     [0, 0, -1, 0, 0]], np.float32) / 8
_RB_AT_G_ROW = np.array([[0, 0, 0.5, 0, 0],
                         [0, -1, 0, -1, 0],
                         [-1, 4, 5, 4, -1],
                         [0, -1, 0, -1, 0],
                         [0, 0, 0.5, 0, 0]], np.float32) / 8
_RB_AT_G_COL = _RB_AT_G_ROW.T.copy()
_RB_AT_BR = np.array([[0, 0, -1.5, 0, 0],
                      [0, 2, 0, 2, 0],
                      [-1.5, 0, 6, 0, -1.5],
                      [0, 2, 0, 2, 0],
                      [0, 0, -1.5, 0, 0]], np.float32) / 8


def demosaic_mhc(m):
    """RGGB mosaic (H,W) -> RGB (H,W,3) with Malvar-He-Cutler interpolation."""
    # REFLECT_101 keeps the Bayer phase at the borders (row -1 mirrors row 1)
    f = lambda k: cv2.filter2D(m, -1, k, borderType=cv2.BORDER_REFLECT_101)
    g, row, col, diag = f(_G_AT_RB), f(_RB_AT_G_ROW), f(_RB_AT_G_COL), f(_RB_AT_BR)
    out = np.empty(m.shape + (3,), np.float32)
    R, G, B = out[..., 0], out[..., 1], out[..., 2]
    s = (slice(0, None, 2), slice(0, None, 2))   # R sites
    R[s], G[s], B[s] = m[s], g[s], diag[s]
    s = (slice(0, None, 2), slice(1, None, 2))   # G in red rows
    R[s], G[s], B[s] = row[s], m[s], col[s]
    s = (slice(1, None, 2), slice(0, None, 2))   # G in blue rows
    R[s], G[s], B[s] = col[s], m[s], row[s]
    s = (slice(1, None, 2), slice(1, None, 2))   # B sites
    R[s], G[s], B[s] = diag[s], g[s], m[s]
    return out


# ---------------------------------------------------------------- colour

def apply_ccm(rgb, ccm):
    return rgb @ np.asarray(ccm, np.float32).T


def gamma(x):
    return np.clip(x, 1e-4, 1) ** (1 / 2.2)


def tone_map(x):
    """Global S-curve (smoothstep): lifts mid-tone contrast, rolls off highlights."""
    x = np.clip(x, 0, 1)
    return 3 * x ** 2 - 2 * x ** 3


def to_ycc(x):
    return x @ RGB2YCC.T


def from_ycc(x):
    return x @ YCC2RGB.T


# ---------------------------------------------------------------- denoise / sharpen

def display_sigma(gain, wb, noise, ccm, level=0.18):
    """Approximate noise std after gain, WB, CCM and gamma at a mid-grey level."""
    shot, read = noise
    w = np.asarray(wb, np.float32)
    var = shot * gain * w * level + read * (gain * w) ** 2    # per channel after WB
    var = (np.asarray(ccm, np.float32) ** 2) @ var              # CCM mixes the channels
    return float(np.sqrt(var.mean()) * level ** (1 / 2.2 - 1) / 2.2)  # gamma slope


def denoise(x, sigma, k_luma=0.6, k_chroma=4.0):
    """Edge-preserving denoising in YCbCr: non-local means on luma, bilateral on chroma."""
    if sigma < 0.5 / 255:
        return x
    ycc = to_ycc(x)
    y8 = np.clip(ycc[..., 0] * 255, 0, 255).astype(np.uint8)
    ycc[..., 0] = cv2.fastNlMeansDenoising(y8, None, h=k_luma * sigma * 255,
                                           templateWindowSize=7, searchWindowSize=21) / 255.0
    for ch in (1, 2):
        ycc[..., ch] = cv2.bilateralFilter(np.ascontiguousarray(ycc[..., ch]), 9, k_chroma * sigma, 4)
    return from_ycc(ycc)


def sharpen(x, amount=0.5, radius=1.0):
    """Unsharp masking on luma only, so no colour fringes are added."""
    ycc = to_ycc(x)
    y = np.ascontiguousarray(ycc[..., 0])
    ycc[..., 0] = y + amount * (y - cv2.GaussianBlur(y, (0, 0), radius))
    return from_ycc(ycc)


def classical_isp(mosaic, meta, k_luma=0.6, k_chroma=4.0, sharp=0.3):
    """Conventional baseline:
    BLC -> gain -> WB -> demosaic -> CCM -> gamma -> denoise -> tone map -> sharpen.
    Denoise and sharpening strength follow the (known or estimated) noise level.
    """
    m = normalize(mosaic, meta)
    gain, wb, noise = resolve_meta(m, meta)
    m = np.minimum(apply_wb(m * gain, wb), 1)   # saturated highlights -> white
    rgb = np.clip(apply_ccm(demosaic_mhc(m), meta['cam2rgb']), 0, 1)
    x = gamma(rgb)
    sigma = display_sigma(gain, wb, noise, meta['cam2rgb'])
    x = denoise(x, sigma, k_luma, k_chroma)
    x = tone_map(x)
    x = sharpen(x, amount=sharp / (1 + 30 * sigma))  # back off when noisy
    return np.clip(x, 0, 1)
