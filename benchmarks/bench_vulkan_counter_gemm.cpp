// Counter-state GEMM benchmark at the MoE trainer's exact shapes.
//
// counter_gemm_db is 29.6% of a training step (Phase D), so iterating on it
// through a 25-step trainer run is too slow and too noisy. This bench issues
// only that kernel, at the shapes the trainer actually uses, and reports the
// GPU time from timestamp queries plus the achieved GFLOP/s against the RX 580
// FP32 peak.
//
//   MOTIFCL_VK_SHADER_INFO=1 bench_vulkan_counter_gemm [reps]
//
// prints VGPR/LDS and the resulting waves/SIMD for the pipeline, which is the
// number this bench exists to move: 2 waves/SIMD of 10 is the current state.
//
// Shapes: the 15M trainer is d_model=288, hidden_exp=192, B=8, T=256, so
// batch=2048 and each expert runs up [2048,288]x[288,192] and
// down [2048,192]x[192,288]. Seven experts, both directions.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

constexpr double kPolarisFp32PeakTflops = 6.17;  // 36 CU * 64 lanes * 2 * 1.34 GHz

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct BenchShape {
    const char* label;
    const char* mode;
    std::size_t batch;
    std::size_t in_features;
    std::size_t out_features;
};

} // namespace

int main(int argc, char** argv) {
    using namespace motifcl;
    if (!probe_vulkan_runtime().available()) {
        std::printf("SKIP bench_vulkan_counter_gemm: %s\n", probe_vulkan_runtime().error.c_str());
        return 77;
    }
    _putenv_s("MOTIFCL_VK_PROFILE", "1");
    const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 15;
    try {
        Backend backend = Backend::create_vulkan();
        auto& runtime = backend.vulkan_runtime();
        runtime.set_gpu_timing_enabled(true);

        const std::size_t C = 3;  // levels -> 2*C-1 = 5 ternary levels, 6-bit codes
        const std::vector<BenchShape> shapes = {
            {"expert_up", "forward", 2048, 288, 192},
            {"expert_down", "forward", 2048, 192, 288},
            {"expert_up", "backward_input", 2048, 288, 192},
            {"expert_down", "backward_input", 2048, 192, 288},
            {"attn_qkv", "forward", 2048, 288, 288},
            {"attn_qkv", "backward_input", 2048, 288, 288},
            // Decode shapes: the fixed ~15 us decode dominates here, which is why
            // CounterStateLinear restricts the dense path to >= 128 rows.
            {"expert_up", "forward", 1, 288, 192},
            {"expert_down", "forward", 1, 192, 288},
        };

        std::printf("%-14s %-15s %-18s %8s %8s %7s %7s %7s %8s %7s\n", "shape", "mode", "batch x in x out", "counter",
                    "GFLOP/s", "%peak", "decode", "gemm", "dec+gemm", "gain");
        for (const auto& shape : shapes) {
            const std::size_t groups_per_row = shape.in_features / 4;
            const std::size_t state_bytes = shape.out_features * groups_per_row * 3;
            const std::size_t scale_bytes = shape.out_features * sizeof(float);
            const std::size_t act_elems = shape.batch * std::max(shape.in_features, shape.out_features);

            std::vector<std::uint8_t> state(state_bytes);
            for (std::size_t i = 0; i < state_bytes; ++i) state[i] = static_cast<std::uint8_t>((i * 37u) & 0xFFu);
            std::vector<float> scale(shape.out_features);
            for (std::size_t i = 0; i < scale.size(); ++i) scale[i] = 0.02f + 0.0001f * static_cast<float>(i % 97);
            std::vector<float> act(act_elems);
            for (std::size_t i = 0; i < act_elems; ++i) act[i] = -0.5f + 0.001f * static_cast<float>((i * 13u) % 977u);

            auto x = runtime.create_buffer(act_elems * sizeof(float), act.data());
            auto state_buf = runtime.create_buffer(state_bytes, state.data());
            auto scale_buf = runtime.create_buffer(scale_bytes, scale.data());
            auto out = runtime.create_buffer(act_elems * sizeof(float));

            const bool forward = std::strcmp(shape.mode, "forward") == 0;
            auto run = [&]() -> VulkanOpResult {
                if (forward) {
                    return run_vulkan_compact_counter_forward_u8(runtime, x, state_buf, scale_buf, out, shape.batch,
                                                                 shape.in_features, shape.out_features, C);
                }
                return run_vulkan_compact_counter_backward_input_u8(runtime, state_buf, scale_buf, x, out,
                                                                    shape.batch, shape.in_features,
                                                                    shape.out_features, C);
            };

            if (!run().success) {
                std::fprintf(stderr, "dispatch failed for %s %s\n", shape.label, shape.mode);
                return 1;
            }
            for (int i = 0; i < 3; ++i) {
                if (!run().success) {
                    std::fprintf(stderr, "warmup dispatch failed for %s %s\n", shape.label, shape.mode);
                    return 1;
                }
            }
            runtime.reset_kernel_profile();
            std::vector<double> gpu_ms;
            gpu_ms.reserve(static_cast<std::size_t>(reps));
            for (int i = 0; i < reps; ++i) {
                if (!run().success) return 1;
                const double us = runtime.last_gpu_time_us();
                if (us > 0.0) gpu_ms.push_back(us / 1000.0);
            }

            // Alternative path: decode the packed state into a dense FP32 weight
            // once, then run the ordinary FP32 GEMM. This exists because the
            // decode-in-GEMM kernel spends 48% of its instructions on address
            // arithmetic (see MOTIFCL_VK_SHADER_ISA); the dense GEMM is a
            // well-tuned kernel that reads 5.4x more weight bytes but issues
            // almost nothing but FMAs. At these shapes the weights are 41 KB,
            // so the extra traffic is irrelevant and only the issue rate counts.
            auto dense_w = runtime.create_buffer(shape.in_features * shape.out_features * sizeof(float));
            std::vector<double> dense_ms;
            dense_ms.reserve(static_cast<std::size_t>(reps));
            std::vector<double> dec_ms;
            dec_ms.reserve(static_cast<std::size_t>(reps));
            std::vector<double> gemm_ms;
            gemm_ms.reserve(static_cast<std::size_t>(reps));
            auto run_decode = [&]() -> bool {
                return run_vulkan_compact_counter_decode_weight(runtime, state_buf, scale_buf, dense_w,
                                                                shape.in_features, shape.out_features, C)
                    .success;
            };
            auto run_gemm = [&]() -> bool {
                VulkanOpResult r;
                if (forward) {
                    // y[batch,out] = x[batch,in] * W[out,in]^T
                    r = run_vulkan_f32_matmul_transpose_b(runtime, x, dense_w, out, shape.batch, shape.in_features,
                                                          shape.out_features);
                } else {
                    // grad_x[batch,in] = grad_out[batch,out] * W[out,in]
                    r = run_vulkan_f32_matmul(runtime, x, dense_w, out, shape.batch, shape.out_features,
                                              shape.in_features);
                }
                return r.success;
            };
            auto run_dense = [&]() -> bool { return run_decode() && run_gemm(); };
            for (int i = 0; i < 3; ++i) {
                if (!run_dense()) {
                    std::fprintf(stderr, "decode+dense warmup failed for %s %s\n", shape.label, shape.mode);
                    return 1;
                }
            }
            for (int i = 0; i < reps; ++i) {
                if (!runtime.batch_begin()) return 1;
                if (!run_dense()) return 1;
                if (!runtime.batch_end().success) return 1;
                const double us = runtime.last_gpu_time_us();
                if (us > 0.0) dense_ms.push_back(us / 1000.0);
            }
            // Decompose: decode alone and the GEMM alone, both unbatched.
            for (int i = 0; i < reps; ++i) {
                if (!run_decode()) return 1;
                const double dec_us = runtime.last_gpu_time_us();
                if (dec_us > 0.0) dec_ms.push_back(dec_us / 1000.0);
            }
            for (int i = 0; i < reps; ++i) {
                if (!run_gemm()) return 1;
                const double gemm_us = runtime.last_gpu_time_us();
                if (gemm_us > 0.0) gemm_ms.push_back(gemm_us / 1000.0);
            }

            std::string dominant;
            unsigned long long best_calls = 0;
            for (const auto& row : runtime.kernel_profile()) {
                if (row.calls > best_calls) {
                    best_calls = row.calls;
                    dominant = row.name;
                }
            }
            const double ms = median(gpu_ms);
            const double dense = median(dense_ms);
            const double gflop = 2.0 * static_cast<double>(shape.batch) * static_cast<double>(shape.in_features) *
                                 static_cast<double>(shape.out_features) / 1e9;
            const double gflops = ms > 0.0 ? gflop / (ms * 1e-3) : 0.0;
            char dims[64];
            std::snprintf(dims, sizeof(dims), "%zu x %zu x %zu", shape.batch, shape.in_features, shape.out_features);
            std::printf("%-14s %-15s %-18s %8.1f %8.1f %6.1f%% %7.1f %7.1f %8.1f %6.2fx\n", shape.label, shape.mode,
                        dims, ms * 1000.0, gflops, 100.0 * gflops / (kPolarisFp32PeakTflops * 1000.0),
                        median(dec_ms) * 1000.0, median(gemm_ms) * 1000.0, dense * 1000.0,
                        dense > 0.0 ? ms / dense : 0.0);
            std::printf("  kernel %s calls=%llu\n", dominant.c_str(), best_calls);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench failed: %s\n", e.what());
        return 1;
    }
}
