# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 151 | 6.56 | 479.1 | 3.47211 GFLOP/s | 3.17285 | 0.33 | PASS |
| matmul_f32 | 256x256x256 | 189.1 | 37.92 | 504.3 | 177.443 GFLOP/s | 2.66684 | 0.33 | PASS |
| matmul_f32 | 512x512x512 | 445.3 | 230.4 | 604.8 | 602.819 GFLOP/s | 1.35819 | 0.33 | PASS |
| matmul_f32_nt | 256x256x256 | 217.3 | 46.56 | 495.9 | 154.415 GFLOP/s | 2.2821 | 0.33 | PASS |
| matmul_f32_nt | 512x512x512 | 584.4 | 347.84 | 725.5 | 459.335 GFLOP/s | 1.24144 | 0.33 | PASS |
| matmul_f32_tn | 256x256x256 | 197.5 | 34.08 | 488.1 | 169.896 GFLOP/s | 2.47139 | 0.33 | PASS |
| matmul_f32_tn | 512x512x512 | 490.7 | 243.84 | 613.5 | 547.046 GFLOP/s | 1.25025 | 0.33 | PASS |
| matmul_f32_m1 | 1x1024x1024 | 343.8 | 124.8 | 570.2 | 6.09992 GFLOP/s | 1.65852 | 0.33 | PASS |
| matmul_f32_m1 | 1x2048x2048 | 478.2 | 248.64 | 621.5 | 17.542 GFLOP/s | 1.29967 | 0.33 | PASS |
| matmul_f32_m1nt | 1x1024x1024 | 203.7 | 46.4 | 883.3 | 10.2953 GFLOP/s | 4.33628 | 0.33 | PASS |
| matmul_f32_m1nt | 1x2048x2048 | 412.8 | 216.96 | 2338.9 | 20.3212 GFLOP/s | 5.66594 | 0.33 | PASS |
| softmax_rows_f32 | 512x1024 | 232.8 | 54.56 | 4394.9 | 18.0168 GB/s | 18.8784 | 0.4 | PASS |
| softmax_rows_bwd_f32 | 512x1024 | 385.4 | 123.52 | unavailable | 16.3245 GB/s | - | 0.4 | NO_BASELINE |
| rmsnorm_f32 | 512x1024 | 283.8 | 49.12 | 415.6 | 14.7791 GB/s | 1.46441 | 0.4 | PASS |
| rmsnorm_bwd_x_f32 | 512x1024 | 301.4 | 110.4 | 364.8 | 20.8741 GB/s | 1.21035 | 0.4 | PASS |
| rmsnorm_bwd_w_f32 | 512x1024 | 665.1 | 213.28 | 637.3 | 6.30628 GB/s | 0.958202 | 0.4 | PASS |
| gelu_f32 | 512x1024 | 183.7 | 19.68 | 306 | 22.8324 GB/s | 1.66576 | 0.6 | PASS |
| gelu_bwd_f32 | 512x1024 | 220.4 | 31.04 | 346.5 | 28.5456 GB/s | 1.57214 | 0.6 | PASS |
| add_f32 | 512x1024 | 203.1 | 29.44 | 370.5 | 30.9771 GB/s | 1.82422 | 0.6 | PASS |
| sub_f32 | 512x1024 | 183.3 | 29.44 | 324.1 | 34.3233 GB/s | 1.76814 | 0.6 | PASS |
| sgd_update_f32 | 512x1024 | 187.1 | 29.28 | 419 | 33.6262 GB/s | 2.23944 | 0.4 | PASS |
| mul_scalar_f32 | 512x1024 | 198.2 | 18.72 | 278.4 | 21.162 GB/s | 1.40464 | 0.6 | PASS |
| swiglu_f32 | 512x2048 | 212.1 | 28.8 | 317.2 | 39.5502 GFLOP/s | 1.49552 | 0.6 | PASS |
| swiglu_bwd_f32 | 512x2048 | 225.3 | 50.72 | 351.2 | 1191.46 GFLOP/s | 1.55881 | 0.6 | PASS |
| embedding_gather_f32_i32 | V4096D64T512 | 167 | 3.52 | 355.4 | 50.2312 GFLOP/s | 2.12814 | 0.6 | PASS |
| rope_f32 | T512h8d64 | 170.3 | 25.76 | 330.4 | 49.2578 GFLOP/s | 1.94011 | 0.4 | PASS |
| gqa_fwd_f32 | q64k64h8kv2d64 | 322.5 | 108.32 | 391.4 | 26.0112 GFLOP/s | 1.21364 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1052.2 | 109.28 | 2283.9 | 23.9173 GFLOP/s | 2.17059 | 0.25 | PASS |
| counter_update_f32 | in1024_out256_N8 | 619.2 | 1.6 | 567.4 | 40.6425 GFLOP/s | 0.916344 | 0.4 | PASS |
| counter_decode_weight_f32 | in1024_out256 | 176.3 | 9.92 | 285.7 | 47.5814 GFLOP/s | 1.62053 | 0.6 | PASS |
| counter_backward_input_f32 | in1024_out256_N8 | 406.1 | 158.24 | 517 | 10.3283 GFLOP/s | 1.27309 | 0.33 | PASS |
| train_step_f32 | block_T16_E64 | 1436.6 | 156.16 | 5062.1 | 6.96088e-07 GB/s | 3.52367 | 0.4 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
