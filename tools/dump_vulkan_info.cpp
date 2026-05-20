#include <iostream>
#include <string>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

const char* device_type_name(std::uint32_t type) {
    switch (type) {
    case 1: return "integrated_gpu";
    case 2: return "discrete_gpu";
    case 3: return "virtual_gpu";
    case 4: return "cpu";
    default: return "other";
    }
}

} // namespace

int main() {
    const auto probe = motifcl::probe_vulkan_runtime();
    std::cout << "loader_found=" << (probe.loader_found ? "true" : "false") << "\n";
    std::cout << "loader=" << probe.loader_path << "\n";
    std::cout << "instance_created=" << (probe.instance_created ? "true" : "false") << "\n";
    std::cout << "api_version=" << motifcl::vulkan_version_string(probe.api_version) << "\n";
    std::cout << "physical_devices=" << probe.physical_device_count << "\n";
    for (std::size_t i = 0; i < probe.devices.size(); ++i) {
        const auto& device = probe.devices[i];
        std::cout << "device[" << i << "].name=" << device.name << "\n";
        std::cout << "device[" << i << "].type=" << device_type_name(device.device_type) << "\n";
        std::cout << "device[" << i << "].vendor_id=0x" << std::hex << device.vendor_id << std::dec << "\n";
        std::cout << "device[" << i << "].device_id=0x" << std::hex << device.device_id << std::dec << "\n";
        std::cout << "device[" << i << "].api_version=" << motifcl::vulkan_version_string(device.api_version) << "\n";
    }
    if (!probe.error.empty()) std::cout << "error=" << probe.error << "\n";
    return probe.loader_found ? 0 : 77;
}
