# Vulkan port perf record

Device (Vulkan): Radeon RX 580 Series  
Device (OpenCL baseline): unavailable  
Methodology: median wall-time of 20 runs after 3 warmup runs; both backends timed as dispatch+wait on the same device. `gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).

| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |
|---|---|---|---|---|---|---|---|---|
| train_step_f32 | block_T16_E64 | 1376.3 | -1 | unavailable | 7.26586e-07 GB/s | - | 0.4 | NO_BASELINE |

Regenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.
