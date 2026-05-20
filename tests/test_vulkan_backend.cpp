#include <iostream>
#include <string>

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

    return ok ? 0 : 1;
}
