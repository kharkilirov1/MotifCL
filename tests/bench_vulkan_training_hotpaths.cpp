// OpenCL-free, paired training-hotpath measurements. Parity lives in tests.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {
using namespace motifcl;
constexpr std::size_t kRows = 2048, kIn = 288, kOut = 384, kC = 11, kVocab = 32000;
constexpr int kWarmup = 50, kIterations = 200;
using Counts = std::map<std::string, std::uint64_t>;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value) != 0) throw std::runtime_error("cannot set benchmark environment");
#else
    if (setenv(name, value, 1) != 0) throw std::runtime_error("cannot set benchmark environment");
#endif
}

void check(const VulkanOpResult& result) {
    if (!result.success) throw std::runtime_error(result.error);
}

VulkanBuffer floats(VulkanRuntime& rt, std::size_t count, std::uint32_t seed, float amplitude) {
    std::vector<float> host(count);
    for (auto& value : host) {
        seed = seed * 1664525u + 1013904223u;
        value = (float((seed >> 8) & 0xffffu) / 32767.5f - 1.0f) * amplitude + 0.125f;
    }
    return rt.create_buffer(count * sizeof(float), host.data());
}

struct Variant {
    const char* name;
    const char* env;
    const char* value;
    std::function<void()> run;
    Counts per_iteration;
};

void batch(VulkanRuntime& rt, const Variant& variant, int iterations) {
    if (!rt.batch_begin()) throw std::runtime_error("benchmark batch_begin failed");
    for (int i = 0; i < iterations; ++i) variant.run();
    check(rt.batch_end());
}

void measure(VulkanRuntime& rt, const char* suite, const Variant& variant, int block) {
    // Reset every selector per block: no inherited environment or previous
    // suite is allowed to silently change either member of the pair.
    set_env("MOTIFCL_VK_COUNTER_WORD_LOAD", "1");
    set_env("MOTIFCL_VK_COUNTER_SWIGLU", "0");
    set_env("MOTIFCL_VK_RMS_DW_SPLIT", "0");
    set_env("MOTIFCL_VK_CE_CACHE", "0");
    set_env("MOTIFCL_VK_CE_STATS_SUBGROUP", "0");
    set_env(variant.env, variant.value);
    batch(rt, variant, kWarmup);
    rt.reset_kernel_profile();
    const auto start = std::chrono::steady_clock::now();
    batch(rt, variant, kIterations);
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / kIterations;
    const auto profile = rt.kernel_profile();
    Counts actual;
    double total_us = 0.0;
    std::ostringstream kernels;
    std::uint64_t calls = 0;
    for (const auto& entry : profile) {
        if (entry.calls == 0) continue;
        if (!std::isfinite(entry.total_us) || entry.total_us < 0.0) {
            throw std::runtime_error("invalid kernel-profile time for " + entry.name);
        }
        if (calls != 0) kernels << ';';
        kernels << entry.name << ':' << entry.calls;
        actual[entry.name] += entry.calls;
        calls += entry.calls;
        total_us += entry.total_us;
    }
    if (rt.caps().timestamps) {
        Counts expected = variant.per_iteration;
        for (auto& entry : expected) entry.second *= kIterations;
        if (actual != expected) {
            throw std::runtime_error(std::string(suite) + '/' + variant.name +
                " profile engagement/count mismatch: " + kernels.str());
        }
        if (!std::isfinite(total_us) || total_us <= 0.0) {
            throw std::runtime_error("timestamp-capable benchmark produced no valid GPU elapsed time");
        }
    }
    std::cout << suite << ',' << variant.name << ',' << block << ',' << kIterations << ',';
    if (rt.caps().timestamps) std::cout << total_us / (1000.0 * kIterations);
    else std::cout << "NA";
    std::cout << ',' << wall_ms << ',';
    if (rt.caps().timestamps) std::cout << calls;
    else std::cout << "NA";
    std::cout << ',' << (kernels.str().empty() ? "unavailable" : kernels.str()) << ','
              << (rt.caps().timestamps ? "ok" : "timestamps_missing") << '\n' << std::flush;
}

void paired(VulkanRuntime& rt, const char* suite, const Variant& a, const Variant& b) {
    int block = 0;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (const auto* variant : {&a, &b, &b, &a}) measure(rt, suite, *variant, block++);
    }
}
} // namespace

int main() {
    using namespace motifcl;
    try {
        set_env("MOTIFCL_VK_PROFILE", "1");
        const auto probe = probe_vulkan_runtime();
        if (!probe.available()) {
            std::cerr << "SKIP bench_vulkan_training_hotpaths: " << probe.error << '\n';
            return 77;
        }
        auto rt = VulkanRuntime::create();
        if (!rt.available()) throw std::runtime_error(rt.error());
        if (!rt.supports_storage_buffer_i8() || rt.caps().max_workgroup_invocations < 128 ||
            rt.caps().max_shared_memory_bytes < 8448) {
            std::cerr << "SKIP bench_vulkan_training_hotpaths: required storage/LDS/workgroup caps missing\n";
            return 77;
        }
        rt.set_gpu_timing_enabled(true);
        const bool wave64 = rt.caps().supports_wave64_arithmetic();
        std::cerr << "device=" << rt.device_name() << " rows=" << kRows << " in=" << kIn
                  << " out=" << kOut << " C=" << kC << " vocab=" << kVocab
                  << " warmup=" << kWarmup << " measured=" << kIterations
                  << " order=ABBAx2 submissions_per_block=1 dispatch_count_scope=measured_block"
                  << " profile_gpu=sum_dispatch_times\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "suite,variant,block,iters,gpu_ms_per_iter,wall_ms_per_iter,dispatch_count,kernels,status\n";

        auto x = floats(rt, kRows * kIn, 0x1234u, 0.8f);
        auto dy = floats(rt, kRows * kOut, 0x5678u, 0.6f);
        std::vector<std::uint8_t> packed(kOut * (kIn / 4) * 3);
        std::uint32_t seed = 0x91743u;
        for (std::size_t i = 0; i < packed.size(); i += 3) {
            std::uint32_t word = 0;
            for (int lane = 0; lane < 4; ++lane) {
                seed = seed * 1664525u + 1013904223u;
                const std::uint32_t code = ((seed >> 24) % 3) * (2 * kC - 1) + (kC - 1);
                word |= code << (lane * 6);
            }
            packed[i] = std::uint8_t(word);
            packed[i + 1] = std::uint8_t(word >> 8);
            packed[i + 2] = std::uint8_t(word >> 16);
        }
        auto state = rt.create_buffer(packed.size(), packed.data());
        auto scale = floats(rt, kOut, 0x88u, 0.025f);
        auto pre = rt.create_buffer(kRows * kOut * sizeof(float));
        auto dx = rt.create_buffer(kRows * kIn * sizeof(float));
        const auto counter_forward = [&] {
            check(run_vulkan_compact_counter_forward_u8(rt, x, state, scale, pre, kRows, kIn, kOut, kC));
        };
        const auto counter_backward = [&] {
            check(run_vulkan_compact_counter_backward_input_u8(rt, state, scale, dy, dx, kRows, kIn, kOut, kC));
        };
        paired(rt, "counter_forward_2048x288x384",
            {"byte", "MOTIFCL_VK_COUNTER_WORD_LOAD", "0", counter_forward, {{"counter_gemm_db", 1}}},
            {"word", "MOTIFCL_VK_COUNTER_WORD_LOAD", "1", counter_forward, {{"counter_gemm_word_db", 1}}});
        paired(rt, "counter_backward_2048x288x384",
            {"byte", "MOTIFCL_VK_COUNTER_WORD_LOAD", "0", counter_backward, {{"counter_gemm_db", 1}}},
            {"word", "MOTIFCL_VK_COUNTER_WORD_LOAD", "1", counter_backward, {{"counter_gemm_word_db", 1}}});

        auto activated = rt.create_buffer(kRows * (kOut / 2) * sizeof(float));
        paired(rt, "counter_swiglu_2048x288x384",
            {"byte_gemm_plus_swiglu", "MOTIFCL_VK_COUNTER_WORD_LOAD", "0", [&] {
                check(run_vulkan_compact_counter_forward_u8(rt, x, state, scale, pre, kRows, kIn, kOut, kC));
                check(run_vulkan_swiglu(rt, pre, activated, kRows, kOut / 2));
            }, {{"counter_gemm_db", 1}, {"swiglu_f32", 1}}},
            {"fused_saved_pre", "MOTIFCL_VK_COUNTER_SWIGLU", "1", [&] {
                check(run_vulkan_compact_counter_swiglu_u8(rt, x, state, scale, pre, activated, kC, kIn, kOut, kRows));
            }, {{"counter_swiglu_db", 1}}});

        auto rms_dy = floats(rt, kRows * kIn, 0x3819u, 0.3f);
        auto row_inv = rt.create_buffer(kRows * sizeof(float));
        auto dw = rt.create_buffer(kIn * sizeof(float));
        const auto rms = [&] {
            check(run_vulkan_rmsnorm_backward_weight(rt, x, rms_dy, row_inv, dw, kRows, kIn, 1e-5f));
        };
        paired(rt, "rms_dw_2048x288",
            {"legacy", "MOTIFCL_VK_RMS_DW_SPLIT", "0", rms,
                {{"rmsnorm_row_inv_f32", 1}, {"rmsnorm_bwd_w_f32", 1}}},
            {"split", "MOTIFCL_VK_RMS_DW_SPLIT", "1", rms,
                {{"rmsnorm_row_inv_f32", 1}, {"rmsnorm_bwd_w_partial_f32", 1}, {"rmsnorm_bwd_w_reduce_f32", 1}}});

        auto logits = floats(rt, kRows * kVocab, 0x17295u, 4.0f);
        std::vector<std::int32_t> target_host(kRows);
        for (std::size_t r = 0; r < kRows; ++r) target_host[r] = std::int32_t((r * 73 + 3) % kVocab);
        auto targets = rt.create_buffer(target_host.size() * sizeof(std::int32_t), target_host.data());
        const float upstream = 1.7f;
        auto go = rt.create_buffer(sizeof(float), &upstream);
        auto partial = rt.create_buffer(kRows * sizeof(float));
        auto stats = rt.create_buffer(2 * kRows * sizeof(float));
        auto loss = rt.create_buffer(sizeof(float));
        auto grad = rt.create_buffer(kRows * kVocab * sizeof(float));
        const std::string mean = wave64 ? "mean_reduce_f32_subgroup" : "mean_reduce_f32";
        paired(rt, "ce_2048x32000",
            {"legacy", "MOTIFCL_VK_CE_CACHE", "0", [&] {
                check(run_vulkan_softmax_cross_entropy(rt, logits, targets, partial, loss, kRows, kVocab));
                check(run_vulkan_softmax_cross_entropy_backward(rt, logits, targets, go, grad, kRows, kVocab));
            }, {{wave64 ? "ce_fwd_rows_f32_subgroup" : "ce_fwd_rows_f32", 1}, {mean, 1}, {"ce_bwd_f32", 1}}},
            {"cached", "MOTIFCL_VK_CE_CACHE", "1", [&] {
                check(run_vulkan_softmax_cross_entropy_cached(rt, logits, targets, partial, stats, loss, kRows, kVocab));
                check(run_vulkan_softmax_cross_entropy_backward_cached(rt, logits, targets, go, stats, grad, kRows, kVocab));
            }, {{"ce_fwd_stats_f32", 1}, {mean, 1}, {"ce_bwd_cached_f32", 1}}});
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL bench_vulkan_training_hotpaths: " << e.what() << '\n';
        return 1;
    }
}
