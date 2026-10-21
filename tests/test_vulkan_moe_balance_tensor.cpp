// MoE-level witness for the GPU-recordable balance-bias update inside an open batch.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/moe7.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

namespace {
constexpr std::size_t kExperts = 7;
bool ok = true;
void expect(bool cond, const char* msg) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", msg);
    std::fflush(stdout);
    if (!cond) ok = false;
}
std::array<float, kExperts> cpu_step(const std::array<float, kExperts>& bias, float load, float gamma,
                                     float clamp_value) {
    const float target = 1.0f / float(kExperts);
    std::array<float, kExperts> out{};
    for (std::size_t i = 0; i < kExperts; ++i) {
        const float delta = target - load;
        const float step = (delta > 0.0f) ? 1.0f : ((delta < 0.0f) ? -1.0f : 0.0f);
        float v = bias[i] + gamma * step;
        v = std::max(-clamp_value, std::min(clamp_value, v));
        out[i] = v;
    }
    return out;
}
bool same7(const std::array<float, kExperts>& a, const std::array<float, kExperts>& b) {
    return std::memcmp(a.data(), b.data(), kExperts * sizeof(float)) == 0;
}
std::array<float, kExperts> read7(motifcl::Tensor& t) {
    auto v = t.to_vector<float>();
    std::array<float, kExperts> out{};
    if (v.size() == kExperts) std::memcpy(out.data(), v.data(), kExperts * sizeof(float));
    return out;
}
int kernel_calls(motifcl::VulkanRuntime& rt) {
    for (const auto& row : rt.kernel_profile())
        if (row.name.find("moe_balance_bias_update_f32") != std::string::npos) return int(row.calls);
    return 0;
}
void set_env(const char* n, const char* v) {
#ifdef _WIN32
    _putenv_s(n, v);
#else
    setenv(n, v, 1);
#endif
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    set_env("MOTIFCL_VK_PROFILE", "1");
    using namespace motifcl;
    try {
        Backend backend = Backend::create_vulkan();
        VulkanRuntime& rt = backend.vulkan_runtime();
        rt.set_gpu_timing_enabled(true);
        nn::MoE7Config cfg;
        cfg.dim = 8;
        cfg.hidden_expert = 4;
        cfg.hidden_shared = 4;
        cfg.balance_clamp = 2.0f;
        const float gamma = 0.01f, clamp_value = cfg.balance_clamp;
        auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
        const int first_id = moe->router_bias.id();
        Tensor old_alias = moe->router_bias;  // shared buffer, must keep its seven floats
        const auto zeros = read7(old_alias);
        // No-op controls: gamma 0 / negative / NaN must not dispatch, reallocate or change the id.
        const int id0 = moe->router_bias.id();
        rt.reset_kernel_profile();
        moe->update_balance_bias(0.0f);
        moe->update_balance_bias(-1.0f);
        moe->update_balance_bias(std::numeric_limits<float>::quiet_NaN());
        expect(moe->router_bias.id() == id0, "gamma 0/negative/NaN are no-ops (no id change)");
        expect(kernel_calls(rt) == 0, "gamma no-ops dispatch nothing");
        // One open batch records two positive updates through the public API.
        expect(rt.capture_begin(), "capture started");
        expect(rt.batch_begin(), "batch opened");
        rt.reset_kernel_profile();
        moe->update_balance_bias(gamma);
        Tensor intermediate_alias = moe->router_bias;
        moe->update_balance_bias(gamma);
        Tensor final_alias = moe->router_bias;
        const int second_id = final_alias.id();
        auto recording = rt.capture_end();
        auto end = rt.batch_end();
        expect(end.success, "batch submitted");
        expect(kernel_calls(rt) == 2, "moe_balance_bias_update_f32 called twice in one batch");
        expect(second_id != first_id, "bias was replaced by a new tensor");
        expect(!final_alias.requires_grad(), "new router bias requires_grad == false");
        const auto want1 = cpu_step(zeros, 0.0f, gamma, clamp_value);
        const auto want2 = cpu_step(want1, 0.0f, gamma, clamp_value);
        expect(same7(read7(intermediate_alias), want1), "intermediate equals CPU step 1 (memcmp 28 bytes)");
        expect(same7(read7(final_alias), want2), "final equals CPU step 2 (memcmp 28 bytes)");
        // Old alias keeps its original seven floats; model can now be destroyed.
        expect(same7(read7(old_alias), zeros), "old alias keeps the initial seven floats");
        moe.reset();
        // Lifetime: recording keeps the buffers; sentinel is overwritten outside any batch.
        std::array<float, kExperts> sentinel{};
        sentinel.fill(123.0f);
        final_alias.storage().vulkan_buffer.upload(sentinel.data(), kExperts * sizeof(float));
        expect(same7(read7(final_alias), sentinel), "sentinel written outside batch");
        auto replayed = rt.replay(recording);
        expect(replayed.success, "replay submitted");
        expect(same7(read7(final_alias), want2), "replay reproduces CPU step 2 (fixed bindings)");
        // Invalid helper failure inside an active batch must throw, not fall back to CPU.
        nn::MoE7Config bad_cfg = cfg;
        bad_cfg.balance_clamp = 0.0f;
        auto bad = std::make_unique<nn::MoE7FFN>(backend, bad_cfg);
        Tensor bad_alias = bad->router_bias;
        const auto bad_before = read7(bad_alias);
        rt.reset_kernel_profile();
        expect(rt.batch_begin(), "invalid-case batch opened");
        bool threw = false;
        try {
            bad->update_balance_bias(gamma);
        } catch (const std::exception&) {
            threw = true;
        }
        expect(threw, "GPU helper failure inside batch throws (no CPU fallback)");
        expect(rt.batch_end().success, "invalid-case batch closed");
        expect(kernel_calls(rt) == 0, "invalid helper dispatched no kernel");
        expect(same7(read7(bad_alias), bad_before), "invalid case left the old bias unchanged");
        // Outside a batch the historical CPU path still works (original contract).
        auto cpu_moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
        Tensor cpu_alias = cpu_moe->router_bias;
        const auto cpu_zero = read7(cpu_alias);
        rt.reset_kernel_profile();
        cpu_moe->update_balance_bias(gamma);
        expect(kernel_calls(rt) == 0, "outside batch the CPU path dispatches nothing");
        expect(same7(read7(cpu_moe->router_bias), cpu_step(cpu_zero, 0.0f, gamma, clamp_value)),
               "outside batch CPU path exact");
        expect(same7(read7(cpu_alias), cpu_zero), "outside batch old alias unchanged");
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (!ok) return 1;
    std::printf("PASS moe balance bias MoE-level witness\n");
    return 0;
}
