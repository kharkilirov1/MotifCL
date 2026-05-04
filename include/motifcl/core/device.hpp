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
    DeviceType type = DeviceType::OpenCL;
    int index = 0;
    std::string name;
};

} // namespace motifcl
