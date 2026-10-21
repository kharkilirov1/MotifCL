#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

float require_close(const std::vector<float>& actual, const std::vector<float>& expected,
                    const std::string& label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    float max_error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
                label + " non-finite value at " + std::to_string(i));
        const float error = std::fabs(actual[i] - expected[i]);
        const float bound = 1.0e-4f + 3.0e-6f * std::fabs(expected[i]);
        require(error <= bound, label + " error " + std::to_string(error) +
                " at " + std::to_string(i) + " exceeds " + std::to_string(bound));
        max_error = std::max(max_error, error);
    }
    return max_error;
}

void verify_comparator_controls() {
    require_close({1.0f, -2.0f}, {1.0f, -2.0f}, "finite comparator anchor");
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), 2.0f}) {
        bool rejected = false;
        try { require_close({bad}, {1.0f}, "bad output control"); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "comparator accepted seeded bad output");
    }
    bool rejected = false;
    try { require_close({1.0f}, {std::numeric_limits<float>::quiet_NaN()}, "bad reference control"); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "comparator accepted non-finite reference");
}

std::vector<float> values(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<float> out(count);
    for (auto& value : out) {
        seed = seed * 1664525u + 1013904223u;
        value = (float((seed >> 8) & 0xffffu) / 32767.5f - 1.0f) * scale;
    }
    return out;
}

struct Reference {
    std::vector<float> partial;
    std::vector<float> weight;
};

Reference reference(const std::vector<float>& x, const std::vector<float>& dy,
                    std::size_t rows, std::size_t cols, float eps, std::size_t splits) {
    std::vector<double> inverse(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = x[row * cols + col];
            squares += value * value;
        }
        inverse[row] = 1.0 / std::sqrt(squares / double(cols) + double(eps));
    }
    Reference out;
    out.partial.resize(splits * cols);
    out.weight.resize(cols);
    for (std::size_t col = 0; col < cols; ++col) {
        double total = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const auto index = row * cols + col;
            total += double(dy[index]) * double(x[index]) * inverse[row];
        }
        out.weight[col] = static_cast<float>(total);
        for (std::size_t split = 0; split < splits; ++split) {
            double part = 0.0;
            const auto begin = rows * split / splits;
            const auto end = rows * (split + 1) / splits;
            for (auto row = begin; row < end; ++row) {
                const auto index = row * cols + col;
                part += double(dy[index]) * double(x[index]) * inverse[row];
            }
            out.partial[split * cols + col] = static_cast<float>(part);
        }
    }
    return out;
}

std::vector<float> download(const motifcl::VulkanBuffer& buffer, std::size_t count) {
    std::vector<float> out(count);
    buffer.download(out.data(), out.size() * sizeof(float));
    return out;
}

void require_result(const motifcl::VulkanOpResult& result, const std::string& label) {
    require(result.success, label + ": " + result.error);
}

void require_profile(motifcl::VulkanRuntime& runtime, bool split, std::uint64_t calls) {
    if (!runtime.caps().timestamps) return;
    std::uint64_t partial = 0, reduce = 0, legacy = 0;
    for (const auto& entry : runtime.kernel_profile()) {
        if (entry.name == "rmsnorm_bwd_w_partial_f32") partial += entry.calls;
        if (entry.name == "rmsnorm_bwd_w_reduce_f32") reduce += entry.calls;
        if (entry.name == "rmsnorm_bwd_w_f32") legacy += entry.calls;
    }
    require(partial == (split ? calls : 0), "partial shader profile engagement mismatch");
    require(reduce == (split ? calls : 0), "reduce shader profile engagement mismatch");
    require(legacy == (split ? 0 : calls), "legacy shader profile engagement mismatch");
}

void run_case(motifcl::VulkanRuntime& runtime, std::size_t rows, std::size_t cols) {
    const float eps = 1.0e-6f;
    auto x = values(rows * cols, 0x18724u, 0.8f);
    auto dy = values(rows * cols, 0x21973u, 0.9f);
    for (auto& value : dy) value *= -1.75f;
    auto xb = runtime.create_buffer(x.size() * sizeof(float), x.data());
    auto gb = runtime.create_buffer(dy.size() * sizeof(float), dy.data());
    auto old_inv = runtime.create_buffer(rows * sizeof(float));
    auto old_out = runtime.create_buffer(cols * sizeof(float));
    set_env("MOTIFCL_VK_RMS_DW_SPLIT", "0");
    runtime.reset_kernel_profile();
    require(runtime.batch_begin(), "legacy batch begin");
    require_result(motifcl::run_vulkan_rmsnorm_backward_weight(
        runtime, xb, gb, old_inv, old_out, rows, cols, eps), "forced old dW");
    require_result(runtime.batch_end(), "legacy batch end");
    require_profile(runtime, false, 1);
    const auto old = download(old_out, cols);
    float max_error = require_close(old, reference(x, dy, rows, cols, eps, 1).weight,
                                    "legacy dW vs independent f64 oracle");

    for (std::size_t splits : {1u, 3u, 7u, 64u}) {
        const auto ref = reference(x, dy, rows, cols, eps, splits);
        auto inv = runtime.create_buffer(rows * sizeof(float));
        auto partial = runtime.create_buffer(splits * cols * sizeof(float));
        auto out = runtime.create_buffer(cols * sizeof(float));
        runtime.reset_kernel_profile();
        require(runtime.batch_begin(), "split batch begin");
        require_result(motifcl::run_vulkan_rmsnorm_backward_weight_split(
            runtime, xb, gb, inv, partial, out, rows, cols, eps, splits), "split dW");
        require_result(runtime.batch_end(), "split batch end");
        require_profile(runtime, true, 1);
        const auto got = download(out, cols);
        const auto parts = download(partial, splits * cols);
        max_error = std::max(max_error, require_close(got, ref.weight, "split dW vs f64"));
        require_close(parts, ref.partial, "split partial layout/ranges vs f64");
        require_close(got, old, "split dW vs forced legacy helper");
        std::vector<float> partial_sum(cols);
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t split = 0; split < splits; ++split) sum += parts[split * cols + col];
            partial_sum[col] = static_cast<float>(sum);
        }
        require_close(got, partial_sum, "final reduction vs CPU sum of device partials");
    }
    std::cout << "RMS dW " << rows << 'x' << cols << " max_abs=" << max_error << '\n';
}

void lifetime_case(motifcl::VulkanRuntime& runtime, bool capture) {
    constexpr std::size_t rows = 257, cols = 65, splits = 64;
    constexpr float eps = 1.0e-5f;
    const auto x = values(rows * cols, capture ? 83u : 91u, 0.9f);
    const auto dy = values(rows * cols, capture ? 97u : 101u, -2.75f);
    auto xb = runtime.create_buffer(x.size() * sizeof(float), x.data());
    auto gb = runtime.create_buffer(dy.size() * sizeof(float), dy.data());
    auto out = runtime.create_buffer(cols * sizeof(float));
    if (capture) require(runtime.capture_begin(), "lifetime capture begin");
    runtime.reset_kernel_profile();
    require(runtime.batch_begin(), "lifetime batch begin");
    {
        auto inv = runtime.create_buffer(rows * sizeof(float));
        auto partial = runtime.create_buffer(splits * cols * sizeof(float));
        require_result(motifcl::run_vulkan_rmsnorm_backward_weight_split(
            runtime, xb, gb, inv, partial, out, rows, cols, eps, splits), "lifetime split dW");
    }
    xb = motifcl::VulkanBuffer();
    gb = motifcl::VulkanBuffer();
    require_result(runtime.batch_end(), "lifetime batch end");
    require_profile(runtime, true, 1);
    const auto expected = reference(x, dy, rows, cols, eps, splits).weight;
    require_close(download(out, cols), expected, "scratch/input lifetime before batch end");
    if (capture) {
        auto recording = runtime.capture_end();
        require(recording.size() == 3, "split capture must contain exactly three dispatches");
        std::vector<float> poison(cols, std::numeric_limits<float>::quiet_NaN());
        out.upload(poison.data(), poison.size() * sizeof(float));
        runtime.reset_kernel_profile();
        require_result(runtime.replay(recording), "split dW replay");
        require_profile(runtime, true, 1);
        require_close(download(out, cols), expected, "capture/replay after all input/scratch handles died");
    }
}

void integrated_case(motifcl::VulkanRuntime& runtime) {
    constexpr std::size_t rows = 257, cols = 65;
    constexpr float eps = 1.0e-6f;
    const auto x = values(rows * cols, 0x3921u, 0.9f);
    const auto dy = values(rows * cols, 0x4719u, -1.25f);
    auto xb = runtime.create_buffer(x.size() * sizeof(float), x.data());
    auto gb = runtime.create_buffer(dy.size() * sizeof(float), dy.data());
    auto inv = runtime.create_buffer(rows * sizeof(float));
    auto out = runtime.create_buffer(cols * sizeof(float));
    set_env("MOTIFCL_VK_RMS_DW_SPLIT", "1");
    runtime.reset_kernel_profile();
    require(runtime.batch_begin(), "integrated split batch begin");
    require_result(motifcl::run_vulkan_rmsnorm_backward_weight(
        runtime, xb, gb, inv, out, rows, cols, eps), "integrated split dW");
    require_result(runtime.batch_end(), "integrated split batch end");
    require_profile(runtime, true, 1);
    require_close(download(out, cols), reference(x, dy, rows, cols, eps, 1).weight,
                  "existing helper selects eligible split path");
}

void validation_controls(motifcl::VulkanRuntime& runtime) {
    auto x = runtime.create_buffer(64 * sizeof(float));
    auto dy = runtime.create_buffer(64 * sizeof(float));
    auto inv = runtime.create_buffer(8 * sizeof(float));
    auto partial = runtime.create_buffer(64 * 8 * sizeof(float));
    auto out = runtime.create_buffer(8 * sizeof(float));
    auto rejected = [&](std::size_t rows, std::size_t cols, float eps, std::size_t splits) {
        const auto result = motifcl::run_vulkan_rmsnorm_backward_weight_split(
            runtime, x, dy, inv, partial, out, rows, cols, eps, splits);
        require(!result.success && !result.error.empty(), "invalid split dW arguments were accepted");
    };
    rejected(0, 8, 1.0e-6f, 4);
    rejected(8, 0, 1.0e-6f, 4);
    rejected(8, 8, -1.0f, 4);
    rejected(8, 8, std::numeric_limits<float>::quiet_NaN(), 4);
    rejected(8, 8, std::numeric_limits<float>::infinity(), 4);
    rejected(8, 8, 1.0e-6f, 0);
    rejected(8, 8, 1.0e-6f, 65);
    rejected(9, 8, 1.0e-6f, 4);
    auto small = runtime.create_buffer(sizeof(float));
    const auto result = motifcl::run_vulkan_rmsnorm_backward_weight_split(
        runtime, x, dy, inv, small, out, 8, 8, 1.0e-6f, 64);
    require(!result.success && !result.error.empty(), "undersized split partial buffer was accepted");
}

} // namespace

int main() {
    try {
        verify_comparator_controls();
        const auto anchor = reference({1.0f, -1.0f}, {2.75f, -1.25f}, 1, 2, 0.0f, 64);
        require_close(anchor.weight, {2.75f, 1.25f}, "analytic RMS oracle anchor");
        set_env("MOTIFCL_VK_PROFILE", "1");
        auto runtime = motifcl::VulkanRuntime::create();
        if (!runtime.available()) {
            std::cout << "Skipping Vulkan RMSNorm split: " << runtime.error() << '\n';
            return 77;
        }
        if (!runtime.caps().timestamps) {
            std::cout << "Skipping exact kernel-profile engagement: timestamps unavailable\n";
        }
        validation_controls(runtime);
        for (std::size_t rows : {1u, 31u, 32u, 33u, 257u, 2048u}) {
            for (std::size_t cols : {1u, 63u, 65u, 288u, 511u}) run_case(runtime, rows, cols);
        }
        integrated_case(runtime);
        lifetime_case(runtime, false);
        lifetime_case(runtime, true);
        std::cout << "PASS Vulkan RMSNorm split: independent f64, old helper, partials, paths, lifetime, replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Vulkan RMSNorm split: " << error.what() << '\n';
        return 1;
    }
}
