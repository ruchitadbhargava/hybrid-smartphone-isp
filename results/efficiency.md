Measured on CPU (4 threads), 1024x1024 output frame.

| Model | Params (M) | Size (MB) | Feature maps / frame (MB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Learned e2e (FP32) | 1.134 | 4.34 | 293 | 846 |
| Hybrid core (FP32) | 1.134 | 4.34 | 293 | 835 |
| Hybrid core (INT8) | 1.134 | 1.14 | 163 | 670 |
| Hybrid-Lite, distilled (FP32) | 0.505 | 1.94 | 203 | 413 |
| Hybrid-Lite, distilled (INT8) | 0.505 | 0.53 | 114 | 447 |

| Full pipeline (RAW in, sRGB out) | Latency (ms) |
|---|---:|
| Classical ISP | 1081 |
| Learned ISP (e2e) | 877 |
| Hybrid ISP | 1067 |
| Hybrid-Lite ISP (FP32) | 658 |
| Hybrid-Lite ISP (INT8) | 657 |
