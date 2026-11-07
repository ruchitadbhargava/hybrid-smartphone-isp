1RUA24CSE0378,1RUA24CSE0457,1RUA24CSE0346,1RUA24CSE0379,1RUA24CSE0394

# Comparative evaluation: P2, Team T67 (Kuck coders)

Intelligent and Computationally Efficient Smartphone ISP. Test set: Kodak24 converted to RAW under 7 scenarios (168 captures). Quality = mean over all captures. Latency and memory: 1024x1024 frame on an Intel i5-10210U laptop CPU (4 threads).

| Approach | PSNR (dB) | SSIM | Delta E 2000 | LPIPS | Parameters | Model size (MB) | Inference latency (ms, full pipeline) | Memory, feature maps per frame (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical ISP (baseline) | 26.21 | 0.711 | 7.26 | 0.343 | none (no learned weights) | - | 1081 | - |
| Learned ISP, end-to-end U-Net | 24.23 | 0.732 | 9.26 | 0.341 | 1.13 M | 4.34 | 877 | 293 |
| Hybrid ISP (proposed) | 27.26 | 0.720 | 6.19 | 0.307 | 1.13 M | 4.34 | 1067 | 293 |
| Hybrid-Lite, distilled (FP32) | 26.13 | 0.677 | 7.48 | 0.383 | 0.50 M | 1.94 | 658 | 203 |
| Hybrid-Lite, distilled (INT8) | 25.98 | 0.670 | 7.68 | 0.386 | 0.50 M | 0.53 | 657 | 114 |

## Per scenario

**PSNR (dB)**

| Approach | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ISP (baseline) | 31.73 | 29.56 | 26.20 | 21.44 | 18.95 | 29.60 | 25.97 |
| Learned ISP, end-to-end U-Net | 27.59 | 26.81 | 25.46 | 22.98 | 21.21 | 18.84 | 26.70 |
| Hybrid ISP (proposed) | 33.49 | 30.60 | 27.47 | 23.02 | 19.55 | 30.53 | 26.17 |
| Hybrid-Lite, distilled (FP32) | 32.39 | 29.85 | 26.47 | 21.18 | 17.28 | 29.80 | 25.95 |
| Hybrid-Lite, distilled (INT8) | 32.18 | 29.72 | 26.39 | 21.11 | 17.03 | 29.68 | 25.75 |

**SSIM**

| Approach | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ISP (baseline) | 0.937 | 0.856 | 0.705 | 0.460 | 0.328 | 0.858 | 0.833 |
| Learned ISP, end-to-end U-Net | 0.919 | 0.854 | 0.742 | 0.543 | 0.419 | 0.804 | 0.841 |
| Hybrid ISP (proposed) | 0.951 | 0.865 | 0.723 | 0.477 | 0.312 | 0.864 | 0.846 |
| Hybrid-Lite, distilled (FP32) | 0.942 | 0.836 | 0.662 | 0.397 | 0.241 | 0.834 | 0.825 |
| Hybrid-Lite, distilled (INT8) | 0.935 | 0.831 | 0.658 | 0.390 | 0.227 | 0.828 | 0.820 |

**Delta E 2000**

| Approach | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ISP (baseline) | 2.41 | 3.84 | 6.44 | 11.44 | 15.03 | 3.82 | 7.83 |
| Learned ISP, end-to-end U-Net | 6.59 | 7.04 | 7.80 | 9.52 | 11.33 | 15.39 | 7.17 |
| Hybrid ISP (proposed) | 2.28 | 3.28 | 4.80 | 8.58 | 13.42 | 3.29 | 7.70 |
| Hybrid-Lite, distilled (FP32) | 2.55 | 3.72 | 5.80 | 11.24 | 17.51 | 3.72 | 7.85 |
| Hybrid-Lite, distilled (INT8) | 2.81 | 3.89 | 5.92 | 11.42 | 17.77 | 3.88 | 8.08 |

**LPIPS**

| Approach | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ISP (baseline) | 0.074 | 0.189 | 0.364 | 0.598 | 0.727 | 0.190 | 0.258 |
| Learned ISP, end-to-end U-Net | 0.129 | 0.198 | 0.328 | 0.540 | 0.648 | 0.327 | 0.218 |
| Hybrid ISP (proposed) | 0.050 | 0.159 | 0.317 | 0.543 | 0.706 | 0.158 | 0.217 |
| Hybrid-Lite, distilled (FP32) | 0.061 | 0.186 | 0.375 | 0.687 | 0.958 | 0.186 | 0.230 |
| Hybrid-Lite, distilled (INT8) | 0.063 | 0.188 | 0.375 | 0.689 | 0.965 | 0.187 | 0.233 |

Scenarios: daylight x1, indoor x4, low_light x16, night x32 and extreme x64 digital gain (x64 is outside the training range); unseen_illuminant uses white-balance gains outside the training range; no_metadata hides the white balance and noise profile so both are estimated.

Reproduce: `python evaluate.py`, `python benchmark.py`, then `python make_eval.py`.
