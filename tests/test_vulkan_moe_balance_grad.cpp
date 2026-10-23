// OpenCL-free witness for the device-side MoE balance gradient.
//
// The host path (nn::MoE7FFN::prepare_balance_gradient) computes
//   p = softmax(logits[n]); dl[n,j] = (E/N)*aux*p[n,j]*(f_j - sum_e f_e p[n,e])
// and uploads the [N,7] result. This test compares the new Vulkan kernel against
// an independent CPU implementation of the same formula on several fixtures, and
// checks that invalid arguments fail without dispatching or mutating the output.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
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

std::vector<float> cpu_reference(const std::vector<float>& logits, std::size_t rows,
                                 const std::array<float, kExperts>& loads, float scale) {
    std::vector<float> dl(rows * kExperts, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * kExperts;
        float mx = logits[base];
        for (std::size_t j = 1; j < kExperts; ++j) mx = std::max(mx, logits[base + j]);
        std::array<float, kExperts> p{};
        float z = 0.0f;
        for (std::size_t j = 0; j < kExperts; ++j) {
            p[j] = std::exp(logits[base + j] - mx);
            z += p[j];
        }
        float load_dot = 0.0f;
        for (std::size_t j = 0; j < kExperts; ++j) {
            p[j] /= z;
            load_dot += loads[j] * p[j];
        }
        for (std::size_t j = 0; j < kExperts; ++j) {
            dl[base + j] = scale * p[j] * (loads[j] - load_dot);
        }
    }
    return dl;
}

int kernel_calls(motifcl::VulkanRuntime& rt) {
    for (const auto& row : rt.kernel_profile()) {
        if (row.name.find("moe_balance_grad_f32") != std::string::npos) return int(row.calls);
    }
    return 0;
}

void fill_logits(std::vector<float>& logits, std::uint32_t seed) {
    std::uint32_t state = seed;
    for (float& value : logits) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<float>((state >> 8) & 0xffffu) / 32767.5f - 1.0f) * 2.5f;
    }
}

bool run_fixture(motifcl::VulkanRuntime& rt, std::size_t rows, std::uint32_t seed,
                 const std::array<float, kExperts>& loads, float aux_weight, const char* label,
                 bool check_engagement) {
    using namespace motifcl;
    std::vector<float> logits(rows * kExperts);
    fill_logits(logits, seed);
    auto logits_buf = rt.create_buffer(logits.size() * sizeof(float), logits.data());
    auto grad_buf = rt.create_buffer(rows * kExperts * sizeof(float));
    const float scale = static_cast<float>(kExperts) / static_cast<float>(rows) * aux_weight;
    rt.reset_kernel_profile();
    const auto run = run_vulkan_moe_balance_gradient(rt, logits_buf, grad_buf, rows, loads, scale);
    if (!run.success) {
        std::printf("FAIL %s: dispatch failed: %s\n", label, run.error.c_str());
        return false;
    }
    if (check_engagement) {
        expect(kernel_calls(rt) > 0, (std::string(label) + ": moe_balance_grad_f32 selected").c_str());
    }
    std::vector<float> got(rows * kExperts);
    grad_buf.download(got.data(), got.size() * sizeof(float));
    const auto want = cpu_reference(logits, rows, loads, scale);
    double max_abs = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    // exp() on the GPU is not IEEE-exact, so the comparison is a tolerance rather
    // than byte equality; the values themselves are ~1e-4..1e-3 in magnitude.
    const bool close = std::isfinite(max_abs) && max_abs < 1.0e-6;
    if (!close) {
        std::printf("  %s worst row=%zu expert=%zu gpu=%.9g cpu=%.9g\n", label, worst / kExperts,
                    worst % kExperts, got[worst], want[worst]);
    }
    expect(close, (std::string(label) + ": matches independent CPU formula").c_str());
    std::printf("  %s rows=%zu max_abs=%.3e\n", label, rows, max_abs);
    return close;
}
} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    _putenv_s("MOTIFCL_VK_PROFILE", "1");
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("SKIP: no Vulkan runtime (%s)\n", probe.error.c_str());
        return 77;
    }
    VulkanRuntime rt = VulkanRuntime::create();
    rt.set_gpu_timing_enabled(true);

    const std::array<float, kExperts> balanced = {1.0f / 7.0f, 1.0f / 7.0f, 1.0f / 7.0f, 1.0f / 7.0f,
                                                  1.0f / 7.0f, 1.0f / 7.0f, 1.0f / 7.0f};
    const std::array<float, kExperts> skewed = {0.30f, 0.05f, 0.02f, 0.0f, 0.28f, 0.14f, 0.21f};
    const std::array<float, kExperts> one_hot = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    run_fixture(rt, 37, 0x1234u, skewed, 0.01f, "N=37 skewed", true);
    run_fixture(rt, 37, 0x9999u, balanced, 0.01f, "N=37 balanced", false);
    run_fixture(rt, 257, 0x5678u, one_hot, 0.02f, "N=257 one-hot load", false);
    run_fixture(rt, 2048, 0x7777u, skewed, 0.01f, "N=2048 trainer rows", false);

    // Invalid arguments: reject before dispatch, leave the output untouched.
    const std::array<float, kExperts> loads = skewed;
    std::vector<float> logits(37 * kExperts, 0.25f);
    auto logits_buf = rt.create_buffer(logits.size() * sizeof(float), logits.data());
    auto grad_buf = rt.create_buffer(logits.size() * sizeof(float));
    auto small_buf = rt.create_buffer(16 * sizeof(float));
    std::vector<float> sentinel(grad_buf.nbytes() / sizeof(float), -7.0f);
    grad_buf.upload(sentinel.data(), sentinel.size() * sizeof(float));
    rt.reset_kernel_profile();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto nan_loads = loads;
    nan_loads[3] = nan;
    struct Bad {
        const char* name;
        bool rejected;
    };
    const float scale = static_cast<float>(kExperts) / 37.0f * 0.01f;
    const std::array<Bad, 7> bads = {{
        {"rows=0 rejected", !run_vulkan_moe_balance_gradient(rt, logits_buf, grad_buf, 0, loads, scale).success},
        {"NaN scale rejected", !run_vulkan_moe_balance_gradient(rt, logits_buf, grad_buf, 37, loads, nan).success},
        {"NaN load rejected", !run_vulkan_moe_balance_gradient(rt, logits_buf, grad_buf, 37, nan_loads, scale).success},
        {"Inf load rejected", !run_vulkan_moe_balance_gradient(rt, logits_buf, grad_buf, 37,
                                                              std::array<float, kExperts>{0, 0, 0, 0, 0, 0,
                                                                                          std::numeric_limits<float>::infinity()},
                                                              scale).success},
        {"aliased buffers rejected",
         !run_vulkan_moe_balance_gradient(rt, logits_buf, logits_buf, 37, loads, scale).success},
        {"too-small logits rejected",
         !run_vulkan_moe_balance_gradient(rt, small_buf, grad_buf, 37, loads, scale).success},
        {"rows overflow rejected",
         !run_vulkan_moe_balance_gradient(rt, logits_buf, grad_buf,
                                          std::numeric_limits<std::uint32_t>::max() + std::size_t{1}, loads, scale)
              .success},
    }};
    for (const auto& bad : bads) expect(bad.rejected, bad.name);
    expect(kernel_calls(rt) == 0, "invalid arguments dispatched no kernel");
    std::vector<float> after(grad_buf.nbytes() / sizeof(float));
    grad_buf.download(after.data(), after.size() * sizeof(float));
    expect(after == sentinel, "invalid arguments mutated nothing (bytes snapshot)");

    if (!ok) return 1;
    std::printf("PASS moe balance gradient GPU witness (4 fixtures, invalid-argument controls)\n");
    return 0;
}
