# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): unavailable  
Methodology: median wall-time of 20 runs after 3 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 197.2 | 12.16 | unavailable | 2.65866 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32 | 256x256x256 | 302.4 | 82.56 | unavailable | 110.96 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32 | 512x512x512 | 614.3 | 394.24 | unavailable | 436.978 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_nt | 256x256x256 | 260.6 | 70.88 | unavailable | 128.758 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_nt | 512x512x512 | 898.7 | 658.88 | unavailable | 298.693 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_tn | 256x256x256 | 231 | 39.52 | unavailable | 145.257 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_tn | 512x512x512 | 507.2 | 287.04 | unavailable | 529.25 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1 | 1x1024x1024 | 421.1 | 215.52 | unavailable | 4.98018 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1 | 1x2048x2048 | 898.4 | 509.92 | unavailable | 9.33728 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1nt | 1x1024x1024 | 288.7 | 80.8 | unavailable | 7.26412 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1nt | 1x2048x2048 | 496.9 | 276.32 | unavailable | 16.8819 GFLOP/s | - | 0.33 | NO_BASELINE |
| softmax_rows_f32 | 512x1024 | 219.2 | 49.28 | unavailable | 19.1346 GB/s | - | 0.4 | NO_BASELINE |
| softmax_rows_bwd_f32 | 512x1024 | 297.2 | 114.72 | unavailable | 21.1691 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 226.6 | 44.96 | unavailable | 18.5097 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_bwd_x_f32 | 512x1024 | 335 | 127.2 | unavailable | 18.7805 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_bwd_w_f32 | 512x1024 | 478.1 | 64.48 | unavailable | 8.77286 GB/s | - | 0.4 | NO_BASELINE |
| gelu_f32 | 512x1024 | 188.2 | 19.52 | unavailable | 22.2864 GB/s | - | 0.6 | NO_BASELINE |
| gelu_bwd_f32 | 512x1024 | 179.4 | 30.24 | unavailable | 35.0694 GB/s | - | 0.6 | NO_BASELINE |
| add_f32 | 512x1024 | 192.2 | 29.28 | unavailable | 32.7339 GB/s | - | 0.6 | NO_BASELINE |
| sub_f32 | 512x1024 | 161.2 | 29.28 | unavailable | 39.0289 GB/s | - | 0.6 | NO_BASELINE |
| sgd_update_f32 | 512x1024 | 217.5 | 32.96 | unavailable | 28.9262 GB/s | - | 0.4 | NO_BASELINE |
| mul_scalar_f32 | 512x1024 | 176.9 | 34.72 | unavailable | 23.71 GB/s | - | 0.6 | NO_BASELINE |
| swiglu_f32 | 512x2048 | 193.2 | 54.88 | unavailable | 43.4193 GFLOP/s | - | 0.6 | NO_BASELINE |
| swiglu_bwd_f32 | 512x2048 | 250.4 | 97.92 | unavailable | 1072.03 GFLOP/s | - | 0.6 | NO_BASELINE |
| embedding_gather_f32_i32 | V4096D64T512 | 141.1 | 4 | unavailable | 59.4515 GFLOP/s | - | 0.6 | NO_BASELINE |
| rope_f32 | T512h8d64 | 166.1 | 22.88 | unavailable | 50.5034 GFLOP/s | - | 0.4 | NO_BASELINE |
| gqa_fwd_f32 | q64k64h8kv2d64 | 277.6 | 103.04 | unavailable | 30.2183 GFLOP/s | - | 0.25 | NO_BASELINE |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1172.5 | 373.76 | unavailable | 21.4634 GFLOP/s | - | 0.25 | NO_BASELINE |
| counter_update_f32 | in1024_out256_N8 | 514.4 | 2.24 | unavailable | 48.9227 GFLOP/s | - | 0.4 | NO_BASELINE |
| counter_decode_weight_f32 | in1024_out256 | 183.8 | 15.04 | unavailable | 45.6399 GFLOP/s | - | 0.6 | NO_BASELINE |
| counter_backward_input_f32 | in1024_out256_N8 | 341.4 | 168.32 | unavailable | 12.2856 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_q8q8_scaled | 512x512x512 | 415.3 | 197.44 | unavailable | 20.1989 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32q4_m1 | 1x1024x1024 | 241.6 | 97.44 | unavailable | 17.3605 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32q4_m1 | 1x2048x2048 | 622.3 | 416.96 | unavailable | 6.74 GFLOP/s | - | 0.33 | NO_BASELINE |
| train_step_f32 | block_T16_E64 | 1293 | 415.84 | unavailable | 7.73395e-07 GB/s | - | 0.4 | NO_BASELINE |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
