#### PSNR (dB) &uarr;

| Method | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 31.73 | 29.56 | 26.20 | 21.44 | 18.95 | 29.60 | 25.97 |
| Learned (e2e) | 27.59 | 26.81 | 25.46 | 22.98 | **21.21** | 18.84 | **26.70** |
| Hybrid | **33.49** | **30.60** | **27.47** | **23.02** | 19.55 | **30.53** | 26.17 |
| Hybrid-Lite | 32.39 | 29.85 | 26.47 | 21.18 | 17.28 | 29.80 | 25.95 |
| Hybrid-Lite INT8 | 32.18 | 29.72 | 26.39 | 21.11 | 17.03 | 29.68 | 25.75 |
| Hybrid (fixed noise map) | 31.99 | 30.60 | 24.37 | 17.25 | 13.94 | - | - |

#### SSIM &uarr;

| Method | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 0.937 | 0.856 | 0.705 | 0.460 | 0.328 | 0.858 | 0.833 |
| Learned (e2e) | 0.919 | 0.854 | **0.742** | **0.543** | **0.419** | 0.804 | 0.841 |
| Hybrid | **0.951** | **0.865** | 0.723 | 0.477 | 0.312 | **0.864** | **0.846** |
| Hybrid-Lite | 0.942 | 0.836 | 0.662 | 0.397 | 0.241 | 0.834 | 0.825 |
| Hybrid-Lite INT8 | 0.935 | 0.831 | 0.658 | 0.390 | 0.227 | 0.828 | 0.820 |
| Hybrid (fixed noise map) | 0.912 | 0.865 | 0.549 | 0.262 | 0.161 | - | - |

#### Colour error &Delta;E2000 &darr;

| Method | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 2.41 | 3.84 | 6.44 | 11.44 | 15.03 | 3.82 | 7.83 |
| Learned (e2e) | 6.59 | 7.04 | 7.80 | 9.52 | **11.33** | 15.39 | **7.17** |
| Hybrid | **2.28** | **3.28** | **4.80** | **8.58** | 13.42 | **3.29** | 7.70 |
| Hybrid-Lite | 2.55 | 3.72 | 5.80 | 11.24 | 17.51 | 3.72 | 7.85 |
| Hybrid-Lite INT8 | 2.81 | 3.89 | 5.92 | 11.42 | 17.77 | 3.88 | 8.08 |
| Hybrid (fixed noise map) | 2.61 | 3.28 | 6.09 | 13.66 | 20.45 | - | - |

#### LPIPS &darr;

| Method | daylight | indoor | low_light | night | extreme | unseen_illuminant | no_metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 0.074 | 0.189 | 0.364 | 0.598 | 0.727 | 0.190 | 0.258 |
| Learned (e2e) | 0.129 | 0.198 | 0.328 | **0.540** | **0.648** | 0.327 | 0.218 |
| Hybrid | **0.050** | **0.159** | **0.317** | 0.543 | 0.706 | **0.158** | **0.217** |
| Hybrid-Lite | 0.061 | 0.186 | 0.375 | 0.687 | 0.958 | 0.186 | 0.230 |
| Hybrid-Lite INT8 | 0.063 | 0.188 | 0.375 | 0.689 | 0.965 | 0.187 | 0.233 |
| Hybrid (fixed noise map) | 0.123 | 0.159 | 0.509 | 1.056 | 1.276 | - | - |

#### Blind noise estimation (estimated / true noise std at mid grey)

| Scenario | median | min | max |
|---|---:|---:|---:|
| daylight | 1.47 | 1.16 | 4.07 |
| indoor | 1.12 | 0.97 | 1.81 |
| low_light | 1.05 | 0.96 | 1.39 |
| night | 1.02 | 0.95 | 1.25 |
| extreme | 1.00 | 0.94 | 1.19 |
