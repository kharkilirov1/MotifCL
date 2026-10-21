// Counter validation uses the production capability predicate and real Vulkan
// buffers. Synthetic caps test eligibility, not simulated GPU execution.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

void verify_capability_eligibility() {
    struct Case { bool arithmetic; std::uint32_t subgroup; bool eligible; };
    const std::vector<Case> cases = {
        {false, 0, false}, {false, 64, false}, {true, 0, false},
        {true, 32, false}, {true, 64, true}, {true, 128, false},
    };
    for (const auto& c : cases) {
        motifcl::VulkanDeviceCaps caps;
        caps.subgroup_arithmetic_compute = c.arithmetic;
        caps.subgroup_size = c.subgroup;
        if (caps.supports_wave64_arithmetic() != c.eligible) {
            throw std::runtime_error("wave64 eligibility mismatch: arithmetic=" +
                std::to_string(c.arithmetic) + " subgroup=" + std::to_string(c.subgroup));
        }
    }
}

std::vector<std::uint8_t> bytes(const motifcl::VulkanBuffer& buffer) {
    std::vector<std::uint8_t> result(buffer.nbytes());
    buffer.download(result.data(), result.size());
    return result;
}

bool profile_empty(const motifcl::VulkanRuntime& runtime) {
    for (const auto& entry : runtime.kernel_profile()) {
        if (entry.calls != 0) return false;
    }
    return true;
}

void verify_update_validation(motifcl::VulkanRuntime& runtime, bool tiled) {
    constexpr std::size_t in = 8, out = 3, rows = 2, C = 3;
    // C=3, ternary=-1, counter=0 gives code 2. Four copies pack as
    // 0x82,0x20,0x08. Non-saturated counters and non-zero ternary weights
    // make the valid controls change packed state AND scale.
    std::vector<std::uint8_t> state_host(out * (in / 4) * 3);
    for (std::size_t i = 0; i < state_host.size(); i += 3) {
        state_host[i] = 0x82;
        state_host[i + 1] = 0x20;
        state_host[i + 2] = 0x08;
    }
    const std::vector<float> scale_host(out, 0.25f), v_host(out, 0.125f);
    const std::vector<float> grad_w_host(out * in, 0.5f);
    const std::vector<float> x_host(rows * in, 0.5f), go_host(rows * out, 0.5f);
    auto state = runtime.create_buffer(state_host.size(), state_host.data());
    auto scale = runtime.create_buffer(out * sizeof(float), scale_host.data());
    auto v = runtime.create_buffer(out * sizeof(float), v_host.data());
    auto grad_w = runtime.create_buffer(grad_w_host.size() * sizeof(float), grad_w_host.data());
    auto x = runtime.create_buffer(x_host.size() * sizeof(float), x_host.data());
    auto grad_out = runtime.create_buffer(go_host.size() * sizeof(float), go_host.data());
    auto new_scale = runtime.create_buffer(out * sizeof(float));
    auto denom = runtime.create_buffer(out * sizeof(float));
    const auto state_before = bytes(state), scale_before = bytes(scale), v_before = bytes(v);
    const char* names[] = {"lr", "lr_scale", "rms_beta", "rms_eps"};
    auto run = [&](const std::array<float, 4>& params) {
        return tiled ? motifcl::run_vulkan_compact_counter_update_apply(
            runtime, state, scale, v, grad_w, in, out, C,
            params[0], params[1], params[2], params[3], 777u)
            : motifcl::run_vulkan_compact_counter_apply_update_fused(
                runtime, state, scale, v, grad_out, x, new_scale, denom,
                C, in, out, rows, params[0], params[1], params[2], params[3], 777u);
    };
    auto require_rejection = [&](std::size_t parameter, float bad) {
        std::array<float, 4> params{1.0f, 0.05f, 0.9f, 1.0e-6f};
        params[parameter] = bad;
        runtime.reset_kernel_profile();
        const auto result = run(params);
        if (result.success || result.error.find(names[parameter]) == std::string::npos) {
            throw std::runtime_error(std::string(tiled ? "tiled" : "legacy") +
                " update failed to reject " + names[parameter] + "=" +
                std::to_string(bad) + ": success=" + std::to_string(result.success) +
                " error=" + result.error);
        }
        if (!profile_empty(runtime) || bytes(state) != state_before ||
            bytes(scale) != scale_before || bytes(v) != v_before) {
            throw std::runtime_error(std::string("invalid ") + names[parameter] +
                " dispatched work or modified persistent counter buffers");
        }
    };
    for (std::size_t parameter = 0; parameter < 4; ++parameter) {
        for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
            require_rejection(parameter, bad);
        }
    }
    require_rejection(2, -0.01f);
    require_rejection(2, 1.01f);
    require_rejection(3, -1.0e-6f);

    // Null twin: rejecting every call must not satisfy this validation test.
    // Both rms_beta endpoints and zero eps are valid, with non-zero gradients.
    for (float beta : {0.0f, 1.0f}) {
        const auto accepted = run({1.0f, 0.05f, beta, 0.0f});
        if (!accepted.success) {
            throw std::runtime_error("valid counter parameters rejected: " + accepted.error);
        }
    }
    if (bytes(state) == state_before || bytes(scale) == scale_before || bytes(v) == v_before) {
        throw std::runtime_error("valid counter update produced no state/scale/RMS delta");
    }
    std::cout << "  " << (tiled ? "tiled" : "legacy")
              << " rejected 15 invalid parameter sets without dispatch/state mutation; "
                 "accepted finite endpoint controls\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        verify_capability_eligibility();
        if (argc == 2 && std::string(argv[1]) == "--capabilities-only") {
            std::cout << "PASS counter wave64 eligibility: 6 synthetic capability cases\n";
            return 0;
        }
#ifdef _WIN32
        _putenv_s("MOTIFCL_VK_PROFILE", "1");
#else
        setenv("MOTIFCL_VK_PROFILE", "1", 1);
#endif
        const auto probe = motifcl::probe_vulkan_runtime();
        if (!probe.available()) {
            std::cout << "SKIP test_vulkan_counter_validation: " << probe.error << '\n';
            return 77;
        }
        auto runtime = motifcl::VulkanRuntime::create();
        if (!runtime.available()) throw std::runtime_error(runtime.error());
        if (!runtime.supports_storage_buffer_i8()) {
            std::cout << "SKIP test_vulkan_counter_validation: storage_buffer_i8 unavailable\n";
            return 77;
        }
        verify_update_validation(runtime, false);
        if (runtime.caps().supports_wave64_arithmetic()) {
            verify_update_validation(runtime, true);
        } else {
            std::cout << "  SKIP wave64-only direct update validation on subgroup="
                      << runtime.caps().subgroup_size << '\n';
        }
        std::cout << "PASS test_vulkan_counter_validation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_counter_validation: " << e.what() << '\n';
        return 1;
    }
}
