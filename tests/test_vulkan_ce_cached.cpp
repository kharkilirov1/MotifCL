#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

#ifdef MOTIFCL_CE_AUTOGRAD_TEST
#include <motifcl/motifcl.hpp>
#include <motifcl/autograd/node.hpp>
#endif

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
                    float atol, float rtol, const std::string& label) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + " size mismatch");
    float max_error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error(label + " non-finite value at " + std::to_string(i));
        }
        const float error = std::fabs(actual[i] - expected[i]);
        if (!(error <= atol + rtol * std::fabs(expected[i]))) {
            throw std::runtime_error(label + " error " + std::to_string(error) + " at " + std::to_string(i));
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

struct Reference {
    std::vector<float> partial;
    std::vector<float> stats;
    std::vector<float> grad;
    float loss = 0.0f;
};

Reference reference_ce(const std::vector<float>& logits, const std::vector<std::int32_t>& targets,
                       std::size_t cols, float upstream) {
    const std::size_t rows = targets.size();
    Reference ref;
    ref.partial.resize(rows);
    ref.stats.resize(2 * rows);
    ref.grad.resize(logits.size());
    double loss_sum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * cols;
        const double maxv = *std::max_element(logits.begin() + base, logits.begin() + base + cols);
        double sum = 0.0;
        for (std::size_t c = 0; c < cols; ++c) sum += std::exp(double(logits[base + c]) - maxv);
        ref.stats[2 * row] = static_cast<float>(maxv);
        ref.stats[2 * row + 1] = static_cast<float>(1.0 / sum);
        const auto target = targets[row];
        const bool valid = target >= 0 && static_cast<std::size_t>(target) < cols;
        // Keep the existing contract: invalid targets have zero forward loss,
        // but backward still returns the softmax term (without a onehot).
        const double row_loss = valid ? std::log(sum) + maxv - logits[base + target] : 0.0;
        ref.partial[row] = static_cast<float>(row_loss);
        loss_sum += row_loss;
        for (std::size_t c = 0; c < cols; ++c) {
            const double prob = std::exp(double(logits[base + c]) - maxv) / sum;
            const double onehot = valid && c == static_cast<std::size_t>(target) ? 1.0 : 0.0;
            ref.grad[base + c] = static_cast<float>(double(upstream) / rows * (prob - onehot));
        }
    }
    ref.loss = static_cast<float>(loss_sum / rows);
    return ref;
}

std::vector<float> make_logits(std::size_t rows, std::size_t cols, bool large_range) {
    std::vector<float> values(rows * cols);
    std::uint32_t state = 0x8ace713u;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t c = 0; c < cols; ++c) {
            state = state * 1664525u + 1013904223u;
            const float unit = float((state >> 8) & 0xffffu) / 32767.5f - 1.0f;
            values[row * cols + c] = unit * (large_range ? 70.0f : 3.0f) + float(row % 7) * 7.0f;
        }
    }
    return values;
}

std::vector<float> download(const motifcl::VulkanBuffer& buffer, std::size_t count) {
    std::vector<float> out(count);
    buffer.download(out.data(), count * sizeof(float));
    return out;
}

std::uint64_t profile_calls(const motifcl::VulkanRuntime& runtime, const std::string& name) {
    for (const auto& entry : runtime.kernel_profile()) {
        if (entry.name == name) return entry.calls;
    }
    return 0;
}

void require_cached_profile(const motifcl::VulkanRuntime& runtime, bool subgroup) {
    if (!runtime.caps().timestamps) return;
    const std::string forward = subgroup ? "ce_fwd_stats_f32_subgroup" : "ce_fwd_stats_f32";
    require(profile_calls(runtime, forward) != 0, forward + " did not engage");
    require(profile_calls(runtime, "ce_bwd_cached_f32") != 0, "cached flat CE backward did not engage");
    require(profile_calls(runtime, "ce_bwd_f32") == 0, "legacy CE backward concealed cached-path failure");
}

void test_case(motifcl::VulkanRuntime& runtime, std::size_t rows, std::size_t cols,
               float upstream, bool invalid_targets, bool large_range, bool subgroup) {
    const auto logits = make_logits(rows, cols, large_range);
    std::vector<std::int32_t> targets(rows);
    for (std::size_t row = 0; row < rows; ++row) targets[row] = std::int32_t((row * 31 + 1) % cols);
    if (invalid_targets) {
        targets[0] = -7;
        if (rows > 1) targets[1] = static_cast<std::int32_t>(cols);
    }
    const auto ref = reference_ce(logits, targets, cols, upstream);
    auto x = runtime.create_buffer(logits.size() * sizeof(float), logits.data());
    auto t = runtime.create_buffer(targets.size() * sizeof(std::int32_t), targets.data());
    auto go = runtime.create_buffer(sizeof(float), &upstream);
    auto partial = runtime.create_buffer(rows * sizeof(float));
    auto stats = runtime.create_buffer(2 * rows * sizeof(float));
    auto loss = runtime.create_buffer(sizeof(float));
    auto grad = runtime.create_buffer(logits.size() * sizeof(float));
    runtime.reset_kernel_profile();
    require(runtime.batch_begin(), "cached CE batch_begin failed");
    const auto fwd = motifcl::run_vulkan_softmax_cross_entropy_cached(
        runtime, x, t, partial, stats, loss, rows, cols);
    require(fwd.success, "cached CE forward: " + fwd.error);
    const auto bwd = motifcl::run_vulkan_softmax_cross_entropy_backward_cached(
        runtime, x, t, go, stats, grad, rows, cols);
    require(bwd.success, "cached CE backward: " + bwd.error);
    const auto submitted = runtime.batch_end();
    require(submitted.success, "cached CE batch_end: " + submitted.error);
    const std::string label = std::to_string(rows) + "x" + std::to_string(cols);
    const float loss_error = require_close(download(loss, 1), {ref.loss}, 4e-5f, 2e-6f, label + " loss");
    require_close(download(partial, rows), ref.partial, 4e-5f, 2e-6f, label + " row losses");
    require_close(download(stats, 2 * rows), ref.stats, 2e-7f, 3e-6f, label + " saved stats");
    const auto actual_grad = download(grad, logits.size());
    const float grad_error = require_close(actual_grad, ref.grad, 2e-7f, 5e-6f, label + " gradient");
    for (std::size_t row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        for (std::size_t c = 0; c < cols; ++c) row_sum += actual_grad[row * cols + c];
        const bool valid = targets[row] >= 0 && static_cast<std::size_t>(targets[row]) < cols;
        const double expected_sum = valid ? 0.0 : double(upstream) / rows;
        require(std::fabs(row_sum - expected_sum) <= 1e-8 + 5e-6 * std::fabs(double(upstream) / rows),
                label + " gradient row sum at " + std::to_string(row));
    }
    require_cached_profile(runtime, subgroup);

    // Independent legacy kernel comparison is secondary to the double CPU oracle.
    auto old_partial = runtime.create_buffer(rows * sizeof(float));
    auto old_loss = runtime.create_buffer(sizeof(float));
    auto old_grad = runtime.create_buffer(logits.size() * sizeof(float));
    require(runtime.batch_begin(), "legacy CE batch_begin failed");
    const auto old_fwd = motifcl::run_vulkan_softmax_cross_entropy(
        runtime, x, t, old_partial, old_loss, rows, cols);
    const auto old_bwd = motifcl::run_vulkan_softmax_cross_entropy_backward(
        runtime, x, t, go, old_grad, rows, cols);
    require(old_fwd.success && old_bwd.success, "legacy CE comparison failed");
    require(runtime.batch_end().success, "legacy CE batch_end failed");
    require_close(download(old_loss, 1), {ref.loss}, 4e-5f, 2e-6f, label + " legacy loss");
    const auto legacy_grad = download(old_grad, logits.size());
    require_close(legacy_grad, ref.grad, 2e-7f, 5e-6f, label + " legacy gradient");
    if (!subgroup) {
        require(std::memcmp(actual_grad.data(), legacy_grad.data(), actual_grad.size() * sizeof(float)) == 0,
                label + " portable cached CE gradient is not bit-exact with legacy backward");
    }
    std::cout << "CE " << label << " subgroup=" << subgroup << " loss_max_abs=" << loss_error
              << " grad_max_abs=" << grad_error << '\n';
}

void test_cached_stats_are_consumed(motifcl::VulkanRuntime& runtime) {
    constexpr std::size_t rows = 2, cols = 37;
    constexpr float upstream = 2.75f;
    const auto logits = make_logits(rows, cols, false);
    const std::vector<std::int32_t> targets = {7, -1};
    auto ref = reference_ce(logits, targets, cols, upstream);
    ref.stats[1] *= 0.5f;
    ref.stats[2] += std::log(2.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t c = 0; c < cols; ++c) {
            const float prob = std::exp(logits[row * cols + c] - ref.stats[2 * row]) * ref.stats[2 * row + 1];
            ref.grad[row * cols + c] = upstream / rows * (prob - (std::int32_t(c) == targets[row] ? 1.0f : 0.0f));
        }
    }
    auto x = runtime.create_buffer(logits.size() * sizeof(float), logits.data());
    auto t = runtime.create_buffer(targets.size() * sizeof(std::int32_t), targets.data());
    auto go = runtime.create_buffer(sizeof(float), &upstream);
    auto stats = runtime.create_buffer(ref.stats.size() * sizeof(float), ref.stats.data());
    auto grad = runtime.create_buffer(logits.size() * sizeof(float));
    const auto run = motifcl::run_vulkan_softmax_cross_entropy_backward_cached(
        runtime, x, t, go, stats, grad, rows, cols);
    require(run.success, "perturbed cached stats backward: " + run.error);
    require_close(download(grad, logits.size()), ref.grad, 2e-7f, 5e-6f, "cached stats consumption");
}

void test_capture_retains_stats(motifcl::VulkanRuntime& runtime) {
    constexpr std::size_t rows = 3, cols = 257;
    float upstream = -1.25f;
    auto logits = make_logits(rows, cols, false);
    const std::vector<std::int32_t> targets = {0, 256, -1};
    auto x = runtime.create_buffer(logits.size() * sizeof(float), logits.data());
    auto t = runtime.create_buffer(targets.size() * sizeof(std::int32_t), targets.data());
    auto go = runtime.create_buffer(sizeof(float), &upstream);
    auto loss = runtime.create_buffer(sizeof(float));
    auto grad = runtime.create_buffer(logits.size() * sizeof(float));
    require(runtime.capture_begin(), "CE capture_begin failed");
    require(runtime.batch_begin(), "CE capture batch_begin failed");
    {
        auto partial = runtime.create_buffer(rows * sizeof(float));
        auto stats = runtime.create_buffer(2 * rows * sizeof(float));
        require(motifcl::run_vulkan_softmax_cross_entropy_cached(
            runtime, x, t, partial, stats, loss, rows, cols).success, "CE captured forward failed");
        require(motifcl::run_vulkan_softmax_cross_entropy_backward_cached(
            runtime, x, t, go, stats, grad, rows, cols).success, "CE captured backward failed");
    }
    // Both partial and stats handles died before submit. Batch/capture must own them.
    const auto recording = runtime.capture_end();
    require(recording.size() == 3, "CE capture must contain stats forward, mean, cached backward");
    require(runtime.batch_end().success, "CE captured batch_end failed");
    auto ref = reference_ce(logits, targets, cols, upstream);
    require_close(download(loss, 1), {ref.loss}, 4e-5f, 2e-6f, "captured CE loss");
    require_close(download(grad, logits.size()), ref.grad, 2e-7f, 5e-6f, "captured CE gradient");
    for (std::size_t i = 0; i < logits.size(); ++i) logits[i] += float(i % 5) * 0.25f;
    upstream = 0.375f;
    x.upload(logits.data(), logits.size() * sizeof(float));
    go.upload(&upstream, sizeof(float));
    require(runtime.replay(recording).success, "CE replay failed");
    ref = reference_ce(logits, targets, cols, upstream);
    require_close(download(loss, 1), {ref.loss}, 4e-5f, 2e-6f, "replayed CE fresh loss");
    require_close(download(grad, logits.size()), ref.grad, 2e-7f, 5e-6f, "replayed CE fresh gradient");
}

void test_cpu_oracle_and_comparator() {
    const auto ref = reference_ce({0.0f, 0.0f, 0.0f, 0.0f}, {0, -1}, 2, 3.0f);
    require_close({ref.loss}, {float(std::log(2.0) / 2)}, 1e-7f, 0.0f, "uniform CPU oracle loss");
    require_close(ref.grad, {-0.75f, 0.75f, 0.75f, 0.75f}, 0.0f, 0.0f, "uniform CPU oracle grad");
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        bool rejected = false;
        try { require_close({bad}, {0.0f}, 1e-5f, 0.0f, "negative control"); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "CE comparator accepted a non-finite value");
    }
}

#ifdef MOTIFCL_CE_AUTOGRAD_TEST
void test_autograd_cache_and_legacy() {
    constexpr std::size_t rows = 4, cols = 257;
    constexpr float upstream = 2.75f;
    auto backend = motifcl::Backend::create_vulkan();
    auto& runtime = backend.vulkan_runtime();
    const auto logits = make_logits(rows, cols, false);
    const std::vector<std::int32_t> targets = {0, 256, -1, 257};
    const auto ref = reference_ce(logits, targets, cols, upstream);
    auto t = motifcl::Tensor::from_cpu(backend, {rows}, motifcl::DType::I32, targets.data());
    auto go = motifcl::Tensor::full(backend, {}, upstream);
    const bool subgroup = runtime.caps().subgroup_arithmetic_compute && runtime.caps().subgroup_size == 64;
    set_env("MOTIFCL_VK_CE_STATS_SUBGROUP", "1");
    for (bool cached : {false, true}) {
        set_env("MOTIFCL_VK_CE_CACHE", cached ? "1" : "0");
        auto x = motifcl::Tensor::from_cpu(backend, {rows, cols}, motifcl::DType::F32, logits.data());
        x.set_requires_grad();
        runtime.reset_kernel_profile();
        require(runtime.batch_begin(), "autograd CE batch_begin failed");
        auto loss = motifcl::softmax_cross_entropy(x, t);
        // Selection belongs to forward; changing the switch afterwards must
        // not discard a cached node's saved buffers or synthesize stale stats.
        set_env("MOTIFCL_VK_CE_CACHE", cached ? "0" : "1");
        loss.backward(go);
        require(runtime.batch_end().success, "autograd CE batch_end failed");
        require_close(loss.to_vector<float>(), {ref.loss}, 4e-5f, 2e-6f, "autograd CE loss");
        require(bool(x.grad()), "CE did not create logits gradient");
        require_close(x.grad()->to_vector<float>(), ref.grad, 2e-7f, 5e-6f, "autograd CE gradient");
        if (cached) {
            require_cached_profile(runtime, subgroup);
        } else if (runtime.caps().timestamps) {
            require(profile_calls(runtime, "ce_bwd_f32") != 0, "forced legacy CE backward did not engage");
            require(profile_calls(runtime, "ce_bwd_cached_f32") == 0, "legacy CE unexpectedly read cached stats");
        }
    }

    auto x = motifcl::Tensor::from_cpu(backend, {rows, cols}, motifcl::DType::F32, logits.data());
    x.set_requires_grad();
    set_env("MOTIFCL_VK_CE_CACHE", "1");
    runtime.reset_kernel_profile();
    {
        motifcl::autograd::NoGradGuard guard;
        const auto loss = motifcl::softmax_cross_entropy(x, t);
        require(!loss.requires_grad() && !loss._grad_fn(), "NoGrad CE retained an autograd node");
        require_close(loss.to_vector<float>(), {ref.loss}, 4e-5f, 2e-6f, "NoGrad CE loss");
    }
    if (runtime.caps().timestamps) {
        require(profile_calls(runtime, "ce_fwd_stats_f32") == 0 &&
                profile_calls(runtime, "ce_fwd_stats_f32_subgroup") == 0,
                "NoGrad CE must not allocate/write unused cached statistics");
    }
    runtime.reset_kernel_profile();
    const auto explicit_grad = motifcl::softmax_cross_entropy_backward_op(x, t, go);
    require_close(explicit_grad.to_vector<float>(), ref.grad, 2e-7f, 5e-6f, "public uncached CE backward");
    if (runtime.caps().timestamps) {
        require(profile_calls(runtime, "ce_bwd_f32") != 0, "public no-stats backward must retain legacy path");
    }
    std::cout << "PASS CE autograd cached/legacy switch, saved-node lifetime, NoGrad, public backward\n";
}
#endif

} // namespace

int main() {
    try {
        test_cpu_oracle_and_comparator();
        set_env("MOTIFCL_VK_PROFILE", "1");
        const auto probe = motifcl::probe_vulkan_runtime();
        if (!probe.available()) {
            std::cout << "SKIP test_vulkan_ce_cached: " << probe.error << '\n';
            return 77;
        }
        auto runtime = motifcl::VulkanRuntime::create();
        require(runtime.available(), "Vulkan runtime: " + runtime.error());
        const bool can_subgroup = runtime.caps().subgroup_arithmetic_compute && runtime.caps().subgroup_size == 64;
        for (bool subgroup : {false, true}) {
            if (subgroup && !can_subgroup) continue;
            set_env("MOTIFCL_VK_CE_STATS_SUBGROUP", subgroup ? "1" : "0");
            test_case(runtime, 1, 1, 3.0f, false, false, subgroup);
            test_case(runtime, 3, 37, 2.75f, false, false, subgroup);
            test_case(runtime, 4, 257, -1.25f, true, false, subgroup);
            test_case(runtime, 3, 32000, 0.375f, false, false, subgroup);
            test_case(runtime, 5, 65, 1.75f, false, true, subgroup);
        }
        // Exceeds 65535 workgroups at local_size=128: exercises the flat
        // kernel's bounded-grid stride rather than an invalid oversized grid.
        set_env("MOTIFCL_VK_CE_STATS_SUBGROUP", "0");
        test_case(runtime, 525, 32000, 0.375f, false, false, false);
        test_cached_stats_are_consumed(runtime);
        test_capture_retains_stats(runtime);
#ifdef MOTIFCL_CE_AUTOGRAD_TEST
        test_autograd_cache_and_legacy();
#endif
        if (!runtime.caps().timestamps) std::cout << "SKIP CE kernel_profile engagement: no timestamps\n";
        if (!can_subgroup) std::cout << "SKIP CE subgroup64 variant: unsupported device\n";
        std::cout << "PASS test_vulkan_ce_cached: CPU oracle, cached/legacy parity, stats consumption, capture/replay\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_ce_cached: " << e.what() << '\n';
        return 1;
    }
}
