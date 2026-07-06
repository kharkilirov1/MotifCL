# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 250.8 | 6.56 | 405.2 | 2.09046 GFLOP/s | 1.61563 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 353.5 | 38.24 | 510.6 | 94.9206 GFLOP/s | 1.44441 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 597.9 | 230.08 | 707.6 | 448.964 GFLOP/s | 1.18348 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 246.4 | 46.56 | 559.6 | 136.179 GFLOP/s | 2.2711 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 586.9 | 347.36 | 929.2 | 457.379 GFLOP/s | 1.58323 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 247.2 | 34.4 | 530.3 | 135.738 GFLOP/s | 2.14523 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 471.2 | 244.48 | 684.2 | 569.685 GFLOP/s | 1.45204 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 355.4 | 124.8 | 567.2 | 5.90082 GFLOP/s | 1.59595 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 473.7 | 248.48 | 765.6 | 17.7087 GFLOP/s | 1.61621 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 281.8 | 47.04 | 866.5 | 7.44199 GFLOP/s | 3.07488 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 491.7 | 217.12 | 2459 | 17.0604 GFLOP/s | 5.00102 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 259 | 52.96 | 4273.2 | 16.1942 GB/s | 16.4988 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 320.7 | 123.2 | unavailable | 19.6179 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 260.8 | 46.72 | 322.3 | 16.0825 GB/s | 1.23581 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 334.8 | 109.44 | 363.4 | 18.7917 GB/s | 1.08542 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 704.5 | 213.76 | 647.2 | 5.95359 GB/s | 0.918666 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 223.3 | 19.84 | 308.7 | 18.7833 GB/s | 1.38245 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 239.8 | 30.24 | 357.1 | 26.2363 GB/s | 1.48916 | 0.6 | PASS |
| add_f32 | 512x1024 | 235.7 | 29.44 | 350.9 | 26.6926 GB/s | 1.48876 | 0.6 | PASS |
| sub_f32 | 512x1024 | 236.2 | 29.44 | 367.5 | 26.6361 GB/s | 1.55588 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 239.6 | 29.28 | 308.8 | 26.2582 GB/s | 1.28881 | 0.4 | PASS |
| mul_scalar_f32 | 512x1024 | 209.5 | 18.72 | 308.7 | 20.0205 GB/s | 1.47351 | 0.6 | PASS |
| swiglu_f32 | 512x2048 | 238.9 | 28.8 | 362.8 | 35.1135 GFLOP/s | 1.51863 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 221.4 | 50.56 | 403.4 | 1212.45 GFLOP/s | 1.82204 | 0.6 | PASS |
| embedding_gather_f32_i32 | V4096D64T512 | 205.7 | 3.52 | 313.9 | 40.7808 GFLOP/s | 1.52601 | 0.6 | PASS |
| rope_f32 | T512h8d64 | 233.3 | 25.76 | 395.3 | 35.9563 GFLOP/s | 1.69438 | 0.4 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 359.6 | 115.52 | 396.2 | 23.3276 GFLOP/s | 1.10178 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1184.1 | 108.64 | 2505.2 | 21.2531 GFLOP/s | 2.1157 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 757.6 | 1.6 | 641.4 | 33.2178 GFLOP/s | 0.846621 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 217.8 | 10.4 | 357.9 | 38.5152 GFLOP/s | 1.64325 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 391.6 | 157.76 | 600.1 | 10.7107 GFLOP/s | 1.53243 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1558.8 | 157.92 | 7285.8 | 6.41519e-07 GB/s | 4.67398 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
