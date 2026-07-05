# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 148.9 | 6.72 | 363.1 | 3.52107 GFLOP/s | 2.43855 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 202.2 | 38.08 | 484 | 165.947 GFLOP/s | 2.39367 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 445.3 | 229.76 | 665.5 | 602.819 GFLOP/s | 1.4945 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 215.6 | 46.56 | 454 | 155.633 GFLOP/s | 2.10575 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 583.8 | 345.92 | 740 | 459.807 GFLOP/s | 1.26756 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 188 | 34.08 | 487.1 | 178.481 GFLOP/s | 2.59096 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 468.2 | 245.92 | 688.8 | 573.335 GFLOP/s | 1.47117 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 324.8 | 124.64 | 476 | 6.45675 GFLOP/s | 1.46552 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 486.3 | 248.16 | 631.7 | 17.2499 GFLOP/s | 1.29899 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 202.6 | 46.4 | 870.7 | 10.3512 GFLOP/s | 4.29763 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 459.8 | 215.84 | 2378.8 | 18.244 GFLOP/s | 5.17355 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 216.5 | 50.56 | 4199.7 | 19.3732 GB/s | 19.3982 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 345.7 | 121.44 | unavailable | 18.1992 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 220.3 | 45.76 | 358.9 | 19.0391 GB/s | 1.62914 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 293.1 | 107.2 | 346 | 21.4652 GB/s | 1.18048 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 619.7 | 212.48 | 601.9 | 6.76828 GB/s | 0.971276 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 186.2 | 19.68 | 381.5 | 22.5258 GB/s | 2.04887 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 197.4 | 30.24 | 325.7 | 31.8716 GB/s | 1.64995 | 0.6 | PASS |
| add_f32 | 512x1024 | 215.7 | 29.44 | 333.8 | 29.1676 GB/s | 1.54752 | 0.6 | PASS |
| sub_f32 | 512x1024 | 176.6 | 29.44 | 302.5 | 35.6255 GB/s | 1.71291 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 193.1 | 29.44 | 425.8 | 32.5813 GB/s | 2.20508 | 0.4 | PASS |
| swiglu_f32 | 512x2048 | 250.6 | 28.96 | 343 | 33.4741 GFLOP/s | 1.36872 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 224.2 | 50.56 | 361.6 | 1197.3 GFLOP/s | 1.61285 | 0.6 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 296.9 | 107.52 | 395.7 | 28.254 GFLOP/s | 1.33277 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1143.4 | 109.12 | 2155.9 | 22.0096 GFLOP/s | 1.88552 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 633.5 | 1.44 | 659.1 | 39.7251 GFLOP/s | 1.04041 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 200.2 | 10.24 | 348.7 | 41.9011 GFLOP/s | 1.74176 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 419.8 | 157.92 | 575.5 | 9.9912 GFLOP/s | 1.37089 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1454.1 | 157.76 | 4827.3 | 6.87711e-07 GB/s | 3.31979 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
