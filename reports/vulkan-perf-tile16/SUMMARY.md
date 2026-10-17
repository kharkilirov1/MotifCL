# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): unavailable  
Methodology: median wall-time of 20 runs after 3 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| matmul_f32 | 64x64x64 | 200.6 | 16 | unavailable | 2.6136 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32 | 256x256x256 | 268.4 | 82.72 | unavailable | 125.017 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32 | 512x512x512 | 625.1 | 392.48 | unavailable | 429.428 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_nt | 256x256x256 | 218.8 | 70.08 | unavailable | 153.357 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_nt | 512x512x512 | 929.1 | 649.28 | unavailable | 288.92 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_tn | 256x256x256 | 217.7 | 39.52 | unavailable | 154.132 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_tn | 512x512x512 | 502.8 | 285.92 | unavailable | 533.881 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1 | 1x1024x1024 | 432.1 | 214.72 | unavailable | 4.8534 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1 | 1x2048x2048 | 700.4 | 499.84 | unavailable | 11.9769 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1nt | 1x1024x1024 | 215.9 | 81.12 | unavailable | 9.71353 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32_m1nt | 1x2048x2048 | 505.3 | 314.4 | unavailable | 16.6012 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32q4_m1 | 1x1024x1024 | 224.6 | 92.8 | unavailable | 18.6746 GFLOP/s | - | 0.33 | NO_BASELINE |
| matmul_f32q4_m1 | 1x2048x2048 | 618.4 | 396.48 | unavailable | 6.78251 GFLOP/s | - | 0.33 | NO_BASELINE |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
