# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): Ellesmere  
Methodology: median wall-time of 50 runs after 5 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| gqa_fwd_f32 | q64k64h8kv2d64 | 304.2 | 82.88 | 413.3 | 27.576 GFLOP/s | 1.35865 | 0.25 | PASS |
| gqa_fwd_bwd_f32 | q64k64h8kv2d64 | 1082.7 | 108.8 | 2165 | 23.2436 GFLOP/s | 1.99963 | 0.25 | PASS |
| matmul_q8q8_scaled | 512x512x512 | 430.3 | 203.84 | 1424.8 | 19.4948 GFLOP/s | 3.31118 | 0.33 | PASS |
| matmul_f32q4_m1 | 1x1024x1024 | 311.6 | 106.24 | 694.1 | 13.4605 GFLOP/s | 2.22754 | 0.33 | PASS |
| matmul_f32q4_m1 | 1x2048x2048 | 625.5 | 415.04 | 1003.4 | 6.70552 GFLOP/s | 1.60416 | 0.33 | PASS |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
