#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include <motifcl/runtime/microkernel.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

int main() {
    using namespace motifcl;

    bool ok = true;
    auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            ok = false;
        }
    };

    const auto result = probe_vulkan_runtime();
    expect(result.physical_device_count == 0 || result.instance_created,
           "Vulkan probe cannot report physical devices without an instance");
    expect(result.error.empty() || !result.available(),
           "Vulkan probe with an error must not report available=true");
    expect(!vulkan_version_string(0).empty(),
           "Vulkan version formatting must be total");

    const auto selection = select_microkernel_backend_from_value(MicrokernelDomain::Matmul, "vulkan");
    expect(selection.requested == MicrokernelBackendKind::Vulkan,
           "Vulkan backend selection must preserve requested backend");
    expect(selection.effective == MicrokernelBackendKind::OpenCL,
           "Vulkan backend must fall back to OpenCL until a compute descriptor lands");
    expect(selection.fallback_to_opencl && !selection.fallback_reason.empty(),
           "Vulkan backend fallback must be explicit and explainable");

    const auto invalid_matmul = run_vulkan_f32_m1_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_matmul.success, "Vulkan parameterized matmul must reject malformed B size");
    expect(!invalid_matmul.error.empty(), "Vulkan parameterized matmul validation failure must explain why");

    if (result.available()) {
        const bool require_vulkan_compute = std::getenv("MOTIFCL_REQUIRE_VULKAN_COMPUTE") != nullptr;
        const auto compute = run_vulkan_smoke_compute();
        if (!compute.success && !require_vulkan_compute) {
            std::cout << "Vulkan compute smoke skipped: " << compute.error << '\n';
            return ok ? 0 : 1;
        }
        expect(compute.success, "Vulkan smoke compute must succeed when strict Vulkan compute is required");
        expect(compute.error.empty(), "Vulkan smoke compute success must not carry an error");
        expect(compute.output == 42.0f, "Vulkan smoke compute must write the shader output");

        const auto matmul = run_vulkan_f32_matmul_smoke();
        expect(matmul.success, "Vulkan f32 matmul smoke must succeed when a Vulkan device is available");
        expect(matmul.error.empty(), "Vulkan f32 matmul smoke success must not carry an error");
        expect(matmul.output.size() == 4, "Vulkan f32 matmul smoke must return four output values");
        if (matmul.output.size() == 4) {
            expect(matmul.output[0] == 90.0f && matmul.output[1] == 100.0f &&
                       matmul.output[2] == 110.0f && matmul.output[3] == 120.0f,
                   "Vulkan f32 matmul smoke output mismatch");
        }

        const std::vector<float> a = {1.0f, 2.0f, 3.0f};
        const std::vector<float> b = {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
        };
        const auto dynamic_matmul = run_vulkan_f32_m1_matmul(a, b, 3, 2);
        expect(dynamic_matmul.success,
               "Vulkan parameterized f32 M=1 matmul must succeed when a Vulkan device is available");
        expect(dynamic_matmul.error.empty(),
               "Vulkan parameterized f32 M=1 matmul success must not carry an error");
        expect(dynamic_matmul.output.size() == 2,
               "Vulkan parameterized f32 M=1 matmul must return N output values");
        if (dynamic_matmul.output.size() == 2) {
            expect(dynamic_matmul.output[0] == 22.0f && dynamic_matmul.output[1] == 28.0f,
                   "Vulkan parameterized f32 M=1 matmul output mismatch");
        }
    }

    return ok ? 0 : 1;
}
