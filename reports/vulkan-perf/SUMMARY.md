# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 146.5 | 6.56 | 397.3 | 3.57876 GFLOP/s | 2.71195 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 250 | 38.08 | 516.3 | 134.218 GFLOP/s | 2.0652 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 457.7 | 229.92 | 650.1 | 586.488 GFLOP/s | 1.42036 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 224.6 | 46.4 | 492.5 | 149.396 GFLOP/s | 2.19279 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 613.5 | 346.4 | 760.1 | 437.548 GFLOP/s | 1.23896 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 181.5 | 34.08 | 492.4 | 184.873 GFLOP/s | 2.71295 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 488.9 | 244.96 | 637.6 | 549.06 GFLOP/s | 1.30415 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 288.9 | 124.8 | 497.5 | 7.25909 GFLOP/s | 1.72205 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 495.2 | 248.16 | 630.1 | 16.9398 GFLOP/s | 1.27242 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 206.7 | 45.92 | 931.8 | 10.1459 GFLOP/s | 4.50798 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 459.9 | 209.12 | 2449.8 | 18.2401 GFLOP/s | 5.32681 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 243.4 | 52.32 | 3331.2 | 17.2321 GB/s | 13.6861 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 378.4 | 124.16 | unavailable | 16.6265 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 233.4 | 47.36 | 460.2 | 17.9705 GB/s | 1.97172 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 331.2 | 108.96 | 438.7 | 18.9959 GB/s | 1.32458 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 715.6 | 213.6 | 644.5 | 5.86124 GB/s | 0.900643 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 172.4 | 19.68 | 416.1 | 24.3289 GB/s | 2.41357 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 225.1 | 30.24 | 410.8 | 27.9496 GB/s | 1.82497 | 0.6 | PASS |
| add_f32 | 512x1024 | 222.3 | 29.44 | 407.4 | 28.3016 GB/s | 1.83266 | 0.6 | PASS |
| sub_f32 | 512x1024 | 256.1 | 29.44 | 377.3 | 24.5664 GB/s | 1.47325 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 183.5 | 29.44 | 355.9 | 34.2859 GB/s | 1.93951 | 0.4 | PASS |
| mul_scalar_f32 | 512x1024 | 166.2 | 18.72 | 274.9 | 25.2365 GB/s | 1.65403 | 0.6 | PASS |
| swiglu_f32 | 512x2048 | 192.4 | 28.96 | 329.3 | 43.5998 GFLOP/s | 1.71154 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 211 | 50.56 | 370.3 | 1272.21 GFLOP/s | 1.75498 | 0.6 | PASS |
| embedding_gather_f32_i32 | V4096D64T512 | 148.3 | 3.68 | 292.1 | 56.5651 GFLOP/s | 1.96966 | 0.6 | PASS |
| rope_f32 | T512h8d64 | 188.7 | 25.76 | 338.5 | 44.4547 GFLOP/s | 1.79385 | 0.4 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 288.5 | 113.6 | 432.7 | 29.0766 GFLOP/s | 1.49983 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 951.1 | 109.12 | 2233.1 | 26.4597 GFLOP/s | 2.34791 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 706.4 | 1.6 | 582 | 35.6255 GFLOP/s | 0.823896 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 173.2 | 9.76 | 238.8 | 48.4331 GFLOP/s | 1.37875 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 371.9 | 157.6 | 519 | 11.278 GFLOP/s | 1.39554 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1459.9 | 156.96 | 4896.9 | 6.84978e-07 GB/s | 3.35427 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
