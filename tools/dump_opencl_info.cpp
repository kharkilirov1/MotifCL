#include <iostream>
#include <motifcl/motifcl.hpp>
int main() {
    try {
        for (const auto& d : motifcl::OpenCLContext::enumerate_devices()) {
            std::cout << d.platform_name << " / " << d.device_name << " / " << d.driver_version << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }
}
