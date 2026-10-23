// GEMM shape benchmark for the Vulkan F32 matmul family.
//
// Measures the kernels the memory-native trainers actually issue, at the exact
// training shapes, and reports GPU time from VK timestamp queries plus the
// achieved TFLOP/s against the RX 580 / Polaris FP32 peak. Use it to compare
// kernel variants without running a whole training step:
//
//   MOTIFCL_VK_MATMUL_VARIANT=auto|base|rb4|db bench_vulkan_gemm_shapes [reps]
//
// The variant name is printed per row, so a single process can be scripted to
// sweep variants by re-reading the environment on each dispatch.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

constexpr double kPolarisFp32PeakTflops = 6.17;  // 36 CU * 64 lanes * 2 * 1.34 GHz

const char* variant_name() {
    const char* env = std::getenv("MOTIFCL_VK_MATMUL_VARIANT");
    return (env && *env) ? env : "auto";
}

struct BenchShape {
    const char* op;
    const char* label;
    std::size_t m;
    std::size_t k;
    std::size_t n;
};

bool dispatch_shape(const BenchShape& shape, motifcl::VulkanRuntime& runtime,
                    const motifcl::VulkanBuffer& a, const motifcl::VulkanBuffer& b,
                    motifcl::VulkanBuffer& c) {
    motifcl::VulkanOpResult result;
    if (std::string(shape.op) == "nn") {
        result = motifcl::run_vulkan_f32_matmul(runtime, a, b, c, shape.m, shape.k, shape.n);
    } else if (std::string(shape.op) == "nt") {
        result = motifcl::run_vulkan_f32_matmul_transpose_b(runtime, a, b, c, shape.m, shape.k, shape.n);
    } else {
        result = motifcl::run_vulkan_f32_matmul_transpose_a(runtime, a, b, c, shape.m, shape.k, shape.n);
    }
    if (!result.success) {
        std::fprintf(stderr, "dispatch failed for %s %s: %s\n", shape.op, shape.label, result.error.c_str());
        return false;
    }
    return true;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    using namespace motifcl;
    if (!probe_vulkan_runtime().available()) {
        std::printf("SKIP bench_vulkan_gemm_shapes: %s\n", probe_vulkan_runtime().error.c_str());
        return 77;
    }
    _putenv_s("MOTIFCL_VK_PROFILE", "1");
    const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 15;
    try {
        Backend backend = Backend::create_vulkan();
        auto& runtime = backend.vulkan_runtime();
        runtime.set_gpu_timing_enabled(true);

        // Training shapes: LM head fwd / dA / dB, expert up/down, attention
        // projections, and the counter grad_w (transposed-A) GEMM.
        const std::vector<BenchShape> shapes = {
            {"nn", "head_fwd 2048x288x32000", 2048, 288, 32000},
            {"nt", "head_dA 2048x32000x288", 2048, 32000, 288},
            {"tn", "head_dB 288x2048x32000", 288, 2048, 32000},
            {"nn", "expert_up 2048x288x384", 2048, 288, 384},
            {"nn", "expert_down 2048x192x288", 2048, 192, 288},
            {"nn", "attn_qkv 2048x288x288", 2048, 288, 288},
            {"tn", "expert_gradw 384x2048x288", 384, 2048, 288},
            {"tn", "attn_gradw 288x2048x288", 288, 2048, 288},
            {"nn", "dense_up 2048x288x768", 2048, 288, 768},
        };

        std::printf("%-28s %-5s %-6s %10s %10s %8s\n", "shape", "op", "variant", "gpu_ms", "TFLOP/s", "%peak");
        for (const auto& shape : shapes) {
            const std::size_t a_elems = (std::string(shape.op) == "tn") ? shape.k * shape.m : shape.m * shape.k;
            const std::size_t b_elems = (std::string(shape.op) == "nn") ? shape.k * shape.n : shape.k * shape.n;

            std::vector<float> av(a_elems), bv(b_elems);
            for (std::size_t i = 0; i < a_elems; ++i) av[i] = -0.4f + 0.001f * static_cast<float>((i * 13u) % 977u);
            for (std::size_t i = 0; i < b_elems; ++i) bv[i] = -0.3f + 0.001f * static_cast<float>((i * 29u) % 977u);
            auto a = runtime.create_buffer(a_elems * sizeof(float), av.data());
            auto b = runtime.create_buffer(b_elems * sizeof(float), bv.data());
            auto c = runtime.create_buffer(shape.m * shape.n * sizeof(float));


            if (!dispatch_shape(shape, runtime, a, b, c)) return 1;
            for (int i = 0; i < 2; ++i) {
                if (!dispatch_shape(shape, runtime, a, b, c)) return 1;
            }
            runtime.reset_kernel_profile();
            std::vector<double> gpu_ms;
            gpu_ms.reserve(static_cast<std::size_t>(reps));
            for (int i = 0; i < reps; ++i) {
                // Time the op inside a one-dispatch batch: that is how the trainer
                // issues it, and it also covers multi-dispatch paths such as the
                // split-K GEMM (a bare dispatch would only time the final reduce).
                if (!runtime.batch_begin()) return 1;
                if (!dispatch_shape(shape, runtime, a, b, c)) return 1;
                if (!runtime.batch_end().success) return 1;
                const double us = runtime.last_gpu_time_us();
                if (us > 0.0) gpu_ms.push_back(us / 1000.0);
            }
            std::string dominant;
            unsigned long long best_calls = 0;
            for (const auto& row : runtime.kernel_profile()) {
                if (row.calls > best_calls) { best_calls = row.calls; dominant = row.name; }
            }
            const double ms = median(gpu_ms);
            const double gflop = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.k) *
                                 static_cast<double>(shape.n) / 1e9;
            const double tflops = ms > 0.0 ? gflop / ms : 0.0;
            std::printf("%-28s %-5s %-6s %10.3f %10.3f %7.1f%%\n", shape.label, shape.op, variant_name(), ms,
                        tflops, 100.0 * tflops / kPolarisFp32PeakTflops);
            std::printf("  kernel %s calls=%llu\n", dominant.c_str(), best_calls);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench failed: %s\n", e.what());
        return 1;
    }
}
