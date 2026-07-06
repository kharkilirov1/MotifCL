# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 221.3 | 6.72 | 385.7 | 2.36913 GFLOP/s | 1.74288 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 267.8 | 38.08 | 573 | 125.297 GFLOP/s | 2.13966 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 463.1 | 229.76 | 703.1 | 579.649 GFLOP/s | 1.51825 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 271.7 | 47.52 | 646.6 | 123.498 GFLOP/s | 2.37983 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 578.1 | 343.36 | 773.4 | 464.341 GFLOP/s | 1.33783 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 255.7 | 34.72 | 524.9 | 131.226 GFLOP/s | 2.0528 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 506 | 244.8 | 701 | 530.505 GFLOP/s | 1.38538 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 379.8 | 124.96 | 532.1 | 5.52173 GFLOP/s | 1.401 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 465.8 | 248.16 | 716.6 | 18.009 GFLOP/s | 1.53843 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 294.1 | 47.36 | 906.2 | 7.13074 GFLOP/s | 3.08126 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 457.3 | 218.24 | 2440.4 | 18.3438 GFLOP/s | 5.33654 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 299.3 | 51.36 | 3740.4 | 14.0137 GB/s | 12.4972 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 354.2 | 115.84 | unavailable | 17.7624 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 285.7 | 46.88 | 356.5 | 14.6808 GB/s | 1.24781 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 355.4 | 105.6 | 434.4 | 17.7025 GB/s | 1.22228 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 792.4 | 226.88 | 643.7 | 5.29317 GB/s | 0.812342 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 233.3 | 19.68 | 325.1 | 17.9782 GB/s | 1.39348 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 276.2 | 30.24 | 346.7 | 22.7786 GB/s | 1.25525 | 0.6 | PASS |
| add_f32 | 512x1024 | 180.9 | 29.28 | 311.4 | 34.7786 GB/s | 1.72139 | 0.6 | PASS |
| sub_f32 | 512x1024 | 217.4 | 29.44 | 345 | 28.9395 GB/s | 1.58694 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 245.7 | 29.44 | 328.5 | 25.6063 GB/s | 1.337 | 0.4 | PASS |
| mul_scalar_f32 | 512x1024 | 237.5 | 18.72 | 284 | 17.6602 GB/s | 1.19579 | 0.6 | PASS |
| swiglu_f32 | 512x2048 | 238 | 28.96 | 403.5 | 35.2463 GFLOP/s | 1.69538 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 265.7 | 50.56 | 395.6 | 1010.3 GFLOP/s | 1.4889 | 0.6 | PASS |
| embedding_gather_f32_i32 | V4096D64T512 | 234.5 | 3.68 | 332.5 | 35.7723 GFLOP/s | 1.41791 | 0.6 | PASS |
| rope_f32 | T512h8d64 | 204.2 | 25.76 | 340.5 | 41.0804 GFLOP/s | 1.66748 | 0.4 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 320.9 | 114.08 | 491.2 | 26.1409 GFLOP/s | 1.53069 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1206.6 | 109.76 | 2504.5 | 20.8568 GFLOP/s | 2.07567 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 748.8 | 1.6 | 666.9 | 33.6082 GFLOP/s | 0.890625 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 179.1 | 9.92 | 309.5 | 46.8376 GFLOP/s | 1.72808 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 369.4 | 157.92 | 597.8 | 11.3544 GFLOP/s | 1.6183 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1605.3 | 156.96 | 6753.7 | 6.22937e-07 GB/s | 4.20713 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
