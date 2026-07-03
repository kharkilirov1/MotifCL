#pragma once

#include <string>

namespace motifcl {

enum class DeviceType {
    CPU,
    OpenCL,
    Vulkan,
    DirectML
};

struct Device {
    // Vulkan is the first-class backend; OpenCL remains as the optional
    // legacy backend (see docs/VULKAN_PORT_PROTOCOL.md).
    DeviceType type = DeviceType::Vulkan;
    int index = 0;
    std::string name;
};

} // namespace motifcl
