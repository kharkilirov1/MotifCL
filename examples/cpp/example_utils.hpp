#pragma once

#include <exception>
#include <iostream>
#include <string>

namespace motifcl_example {

inline bool is_opencl_unavailable(const std::exception& e) {
    const std::string msg = e.what();
    return msg.find("no OpenCL platforms found") != std::string::npos ||
           msg.find("no OpenCL GPU or CPU device found") != std::string::npos ||
           msg.find("CL_PLATFORM_NOT_FOUND_KHR") != std::string::npos;
}

inline int handle_exception(const std::exception& e, const char* context) {
    if (is_opencl_unavailable(e)) {
        std::cerr << "OpenCL unavailable; skipping " << context << ": " << e.what() << "\n";
        return 0;
    }
    std::cerr << context << " failed: " << e.what() << "\n";
    return 1;
}

} // namespace motifcl_example
