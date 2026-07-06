# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 204.3 | 6.56 | 377.6 | 2.56627 GFLOP/s | 1.84826 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 230 | 38.08 | 629.4 | 145.889 GFLOP/s | 2.73652 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 429 | 229.92 | 619 | 625.724 GFLOP/s | 1.44289 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 253.1 | 50.24 | 519.6 | 132.574 GFLOP/s | 2.05294 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 572.5 | 350.4 | 711 | 468.883 GFLOP/s | 1.24192 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 249.6 | 34.56 | 433.1 | 134.433 GFLOP/s | 1.73518 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 384.9 | 244.32 | 674.2 | 697.416 GFLOP/s | 1.75162 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 354.7 | 124.8 | 531.7 | 5.91247 GFLOP/s | 1.49901 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 470.3 | 247.68 | 657.9 | 17.8367 GFLOP/s | 1.39889 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 272.5 | 48 | 862.7 | 7.69597 GFLOP/s | 3.16587 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 454.3 | 212.48 | 2412.5 | 18.4649 GFLOP/s | 5.31037 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 265.7 | 52.8 | 3372.4 | 15.7859 GB/s | 12.6925 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 314.7 | 125.28 | unavailable | 19.9919 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 247.5 | 46.88 | 319.8 | 16.9467 GB/s | 1.29212 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 339.7 | 108.48 | 341 | 18.5206 GB/s | 1.00383 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 676.9 | 213.92 | 590.4 | 6.19634 GB/s | 0.872212 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 187.7 | 19.52 | 284.8 | 22.3458 GB/s | 1.51731 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 226.4 | 30.24 | 317.6 | 27.7891 GB/s | 1.40283 | 0.6 | PASS |
| add_f32 | 512x1024 | 240.4 | 29.44 | 334.8 | 26.1708 GB/s | 1.39268 | 0.6 | PASS |
| sub_f32 | 512x1024 | 231 | 29.28 | 388.5 | 27.2357 GB/s | 1.68182 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 225.7 | 29.44 | 366 | 27.8753 GB/s | 1.62162 | 0.4 | PASS |
| mul_scalar_f32 | 512x1024 | 223.8 | 18.72 | 294.2 | 18.7413 GB/s | 1.31457 | 0.6 | PASS |
| swiglu_f32 | 512x2048 | 242.1 | 28.96 | 334.6 | 34.6494 GFLOP/s | 1.38207 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 248.1 | 50.56 | 376.3 | 1081.96 GFLOP/s | 1.51673 | 0.6 | PASS |
| embedding_gather_f32_i32 | V4096D64T512 | 200.3 | 3.68 | 335.1 | 41.8802 GFLOP/s | 1.67299 | 0.6 | PASS |
| rope_f32 | T512h8d64 | 245.6 | 25.76 | 379.8 | 34.1556 GFLOP/s | 1.54642 | 0.4 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 338.7 | 115.2 | 480.3 | 24.7671 GFLOP/s | 1.41807 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1264.1 | 109.44 | 2504.3 | 19.9081 GFLOP/s | 1.98109 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 747 | 1.6 | 588.5 | 33.6892 GFLOP/s | 0.787818 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 211.5 | 10.24 | 358.6 | 39.6624 GFLOP/s | 1.69551 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 368.7 | 157.6 | 591.4 | 11.3759 GFLOP/s | 1.60401 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1403.2 | 157.76 | 6533.4 | 7.12657e-07 GB/s | 4.65607 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
