#include <iostream>
#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        auto devices = motifcl::OpenCLContext::enumerate_devices();
        for (std::size_t i = 0; i < devices.size(); ++i) {
            const auto& d = devices[i];
            std::cout << "Device " << i << "\n"
                      << "  platform: " << d.platform_name << " (" << d.platform_vendor << ")\n"
                      << "  device:   " << d.device_name << " (" << d.device_vendor << ")\n"
                      << "  driver:   " << d.driver_version << "\n"
                      << "  version:  " << d.device_version << "\n"
                      << "  CU:       " << d.compute_units << "\n"
                      << "  global:   " << (d.global_mem_size / (1024 * 1024)) << " MiB\n"
                      << "  local:    " << d.local_mem_size << " bytes\n"
                      << "  max WG:   " << d.max_work_group_size << "\n";
        }
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "01_device_info");
    }
}
