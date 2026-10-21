// OpenCL-free witness for the GPU-recordable MoE balance-bias update.
// Links only motifcl_vulkan_runtime: no Tensor/Backend/OpenCL.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {
constexpr std::size_t kExperts = 7;
bool ok = true;
void expect(bool cond, const char* msg) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", msg);
    std::fflush(stdout);
    if (!cond) ok = false;
}
// Independent CPU copy of the established formula (not a call into the new helper).
std::array<float, kExperts> cpu_reference(const std::array<float, kExperts>& old_bias,
                                          const std::array<float, kExperts>& loads, float gamma, float clamp_value) {
    const float target = 1.0f / float(kExperts);
    std::array<float, kExperts> out{};
    for (std::size_t i = 0; i < kExperts; ++i) {
        const float delta = target - loads[i];
        const float step = (delta > 0.0f) ? 1.0f : ((delta < 0.0f) ? -1.0f : 0.0f);
        float v = old_bias[i] + gamma * step;
        v = std::max(-clamp_value, std::min(clamp_value, v));
        out[i] = v;
    }
    return out;
}
std::vector<float> download(motifcl::VulkanBuffer& b) {
    std::vector<float> v(kExperts);
    b.download(v.data(), kExperts * sizeof(float));
    return v;
}
int kernel_calls(motifcl::VulkanRuntime& rt) {
    for (const auto& row : rt.kernel_profile())
        if (row.name.find("moe_balance_bias_update_f32") != std::string::npos) return int(row.calls);
    return 0;
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    _putenv_s("MOTIFCL_VK_PROFILE", "1");
    using namespace motifcl;
    auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("SKIP: no Vulkan runtime (%s)\n", probe.error.c_str());
        return 77;
    }
    VulkanRuntime rt = VulkanRuntime::create();
    rt.set_gpu_timing_enabled(true);

    const float clamp_value = 0.05f, gamma = 0.01f;
    // loads below / above / exactly target(=1/7); old bias near +/- clamp edges and 0
    const std::array<float, kExperts> loads = {0.05f, 0.30f, 1.0f / 7.0f, 0.0f, 0.5f, 0.14f, 0.1428f};
    const std::array<float, kExperts> old_bias = {0.0f, -0.049f, 0.02f, 0.049f, -0.02f, 0.0f, -0.05f};
    auto old_buf = rt.create_buffer(kExperts * sizeof(float), old_bias.data());
    auto new_buf = rt.create_buffer(kExperts * sizeof(float), old_bias.data());

    // Missing-path control (RED): closed batch must fail with zero kernel calls and no mutation.
    auto closed = run_vulkan_moe_balance_bias_update(rt, old_buf, new_buf, loads, gamma, clamp_value);
    expect(!closed.success, "closed batch rejected (control)");
    expect(kernel_calls(rt) == 0, "closed batch dispatched no kernel");
    expect(download(new_buf) == std::vector<float>(old_bias.begin(), old_bias.end()), "closed batch mutated nothing");

    // GREEN: two sequential updates inside one open batch, exact vs independent CPU formula.
    expect(rt.batch_begin(), "batch opened");
    rt.reset_kernel_profile();
    auto r1 = run_vulkan_moe_balance_bias_update(rt, old_buf, new_buf, loads, gamma, clamp_value);
    expect(r1.success, "first update recorded");
    // deferred execution: the batch owns the write, so an immediate host read of the
    // written buffer must be refused by the runtime (not silently observed early).
    bool deferred = false;
    try {
        (void)download(new_buf);
    } catch (const std::exception&) {
        deferred = true;
    }
    expect(deferred, "written buffer read is deferred until batch_end");
    auto end1 = rt.batch_end();
    expect(end1.success, "first batch submitted");
    expect(kernel_calls(rt) > 0, "moe_balance_bias_update_f32 selected (profile name after submit)");
    const auto want1 = cpu_reference(old_bias, loads, gamma, clamp_value);
    { const auto got = download(new_buf); bool same = got == std::vector<float>(want1.begin(), want1.end());
      if (!same) { for (std::size_t i = 0; i < kExperts; ++i) if (got[i] != want1[i]) std::printf("  update1 diff[%zu] gpu=%.9g cpu=%.9g\n", i, got[i], want1[i]); }
      expect(same, "update1 exact 7 floats vs independent CPU"); }

    const auto second_loads = std::array<float, kExperts>{0.2f, 0.1f, 1.0f / 7.0f, 0.1428f, 0.0f, 0.4f, 0.05f};
    expect(rt.batch_begin(), "second batch opened");
    auto r2 = run_vulkan_moe_balance_bias_update(rt, new_buf, old_buf, second_loads, gamma, clamp_value);
    expect(r2.success, "second update recorded");
    auto end2 = rt.batch_end();
    expect(end2.success, "second batch submitted");
    const auto want2 = cpu_reference(want1, second_loads, gamma, clamp_value);
    { const auto got = download(old_buf); bool same = got == std::vector<float>(want2.begin(), want2.end());
      if (!same) { for (std::size_t i = 0; i < kExperts; ++i) if (got[i] != want2[i]) std::printf("  update2 diff[%zu] gpu=%.9g cpu=%.9g\n", i, got[i], want2[i]); }
      expect(same, "update2 exact 7 floats vs independent CPU"); }

    // Invalid arguments must fail before any dispatch/state change.
    const auto before_calls = kernel_calls(rt);
    (void)before_calls;
    const auto snapshot_old = download(old_buf);
    const auto snapshot_new = download(new_buf);
    const auto sentinel = std::vector<float>(old_bias.begin(), old_bias.end());
    auto small = rt.create_buffer(2 * sizeof(float), old_bias.data());
    rt.reset_kernel_profile();
    expect(rt.batch_begin(), "invalid-arg batch opened");
    struct Bad { const char* name; bool cond; };
    const std::array<float, kExperts> nan_loads = [] {
        std::array<float, kExperts> v{};
        v[3] = std::numeric_limits<float>::quiet_NaN();
        return v;
    }();
    const std::vector<Bad> bads = {
        {"gamma=0 rejected", !run_vulkan_moe_balance_bias_update(rt, old_buf, new_buf, loads, 0.0f, clamp_value).success},
        {"gamma=NaN rejected", !run_vulkan_moe_balance_bias_update(rt, old_buf, new_buf, loads,
                                                                   std::numeric_limits<float>::quiet_NaN(), clamp_value).success},
        {"clamp<=0 rejected", !run_vulkan_moe_balance_bias_update(rt, old_buf, new_buf, loads, gamma, 0.0f).success},
        {"NaN load rejected", !run_vulkan_moe_balance_bias_update(rt, old_buf, new_buf, nan_loads, gamma, clamp_value).success},
        {"too-small buffer rejected", !run_vulkan_moe_balance_bias_update(rt, small, new_buf, loads, gamma, clamp_value).success},
        {"same wrapper rejected", !run_vulkan_moe_balance_bias_update(rt, old_buf, old_buf, loads, gamma, clamp_value).success},
    };
    for (const auto& bad : bads) expect(bad.cond, bad.name);
    expect(kernel_calls(rt) == 0, "invalid arguments dispatched no kernel");
    auto end3 = rt.batch_end();
    expect(end3.success, "invalid-arg batch submitted empty");
    expect(kernel_calls(rt) == 0, "still zero kernel calls after empty batch");
    expect(download(old_buf) == snapshot_old && download(new_buf) == snapshot_new, "invalid arguments mutated nothing (bytes snapshot)");
    (void)sentinel;

    if (!ok) return 1;
    std::printf("PASS moe balance bias GPU witness (closed-batch control, 2 exact updates, invalid args)\n");
    return 0;
}
