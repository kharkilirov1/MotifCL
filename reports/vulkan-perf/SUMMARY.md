# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 170.4 | 6.56 | 342.5 | 3.07681 GFLOP/s | 2.00998 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 212.9 | 37.92 | 501 | 157.607 GFLOP/s | 2.35322 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 429.5 | 229.92 | 598 | 624.995 GFLOP/s | 1.39232 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 216.5 | 46.56 | 488.1 | 154.986 GFLOP/s | 2.2545 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 629.5 | 343.2 | 758.5 | 426.426 GFLOP/s | 1.20492 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 192.8 | 34.08 | 492.9 | 174.038 GFLOP/s | 2.55654 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 467.5 | 244.16 | 670.8 | 574.193 GFLOP/s | 1.43487 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 313.8 | 124.8 | 490.7 | 6.68308 GFLOP/s | 1.56373 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 477.2 | 248 | 673.9 | 17.5788 GFLOP/s | 1.4122 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 229.9 | 47.84 | 849.6 | 9.12202 GFLOP/s | 3.69552 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 428.8 | 209.6 | 2347.9 | 19.563 GFLOP/s | 5.47551 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 240.2 | 52.16 | 3297.2 | 17.4617 GB/s | 13.7269 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 346.5 | 124.8 | unavailable | 18.1572 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 222.7 | 46.4 | 339.6 | 18.8339 GB/s | 1.52492 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 296.8 | 108.16 | 340.4 | 21.1976 GB/s | 1.1469 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 626.2 | 213.44 | 589.3 | 6.69803 GB/s | 0.941073 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 191 | 19.52 | 291 | 21.9597 GB/s | 1.52356 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 200.4 | 30.4 | 317.4 | 31.3945 GB/s | 1.58383 | 0.6 | PASS |
| add_f32 | 512x1024 | 231.2 | 29.44 | 319.9 | 27.2122 GB/s | 1.38365 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 191.8 | 29.44 | 302.7 | 32.8022 GB/s | 1.57821 | 0.4 | PASS |
| swiglu_f32 | 512x2048 | 208.5 | 28.96 | 325.4 | 40.2331 GFLOP/s | 1.56067 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 233.9 | 50.72 | 322 | 1147.65 GFLOP/s | 1.37666 | 0.6 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 277.8 | 106.56 | 403.7 | 30.1966 GFLOP/s | 1.4532 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1053 | 109.12 | 2196.5 | 23.8992 GFLOP/s | 2.08594 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 640.1 | 1.44 | 566.7 | 39.3155 GFLOP/s | 0.88533 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 189.7 | 10.88 | 263.8 | 44.2204 GFLOP/s | 1.39062 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 378.2 | 159.2 | 542.5 | 11.0902 GFLOP/s | 1.43443 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1384.6 | 159.2 | 5246.8 | 7.2223e-07 GB/s | 3.7894 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
