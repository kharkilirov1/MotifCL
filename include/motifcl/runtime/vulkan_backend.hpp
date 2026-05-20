#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace motifcl {

// Minimal Vulkan loader probe for the native-GPU backend line.
//
// This intentionally does not expose Vulkan SDK types. The implementation uses
// the platform Vulkan loader dynamically, so the normal OpenCL build and CI do
// not gain a hard Vulkan SDK/link dependency while the runtime can still detect
// whether a Vulkan GPU path is viable on a user's machine.
struct VulkanPhysicalDeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t device_type = 0;
    std::uint32_t api_version = 0;
    std::uint32_t driver_version = 0;
};

struct VulkanProbeResult {
    bool loader_found = false;
    bool instance_created = false;
    std::uint32_t api_version = 0;
    std::uint32_t physical_device_count = 0;
    std::string loader_path;
    std::string error;
    std::vector<VulkanPhysicalDeviceInfo> devices;

    bool available() const {
        return loader_found && instance_created && physical_device_count > 0 && error.empty();
    }
};

struct VulkanSmokeComputeResult {
    bool success = false;
    float output = 0.0f;
    std::string device_name;
    std::string error;
};

VulkanProbeResult probe_vulkan_runtime();
VulkanSmokeComputeResult run_vulkan_smoke_compute();
std::string vulkan_version_string(std::uint32_t version);

} // namespace motifcl
