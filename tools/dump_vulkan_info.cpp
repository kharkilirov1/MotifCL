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
    if (probe.available()) {
        const auto smoke = motifcl::run_vulkan_smoke_compute();
        std::cout << "smoke_compute_success=" << (smoke.success ? "true" : "false") << "\n";
        std::cout << "smoke_compute_output=" << smoke.output << "\n";
        std::cout << "smoke_compute_device=" << smoke.device_name << "\n";
        if (!smoke.error.empty()) std::cout << "smoke_compute_error=" << smoke.error << "\n";
        const auto matmul = motifcl::run_vulkan_f32_matmul_smoke();
        std::cout << "f32_matmul_smoke_success=" << (matmul.success ? "true" : "false") << "\n";
        std::cout << "f32_matmul_smoke_output=";
        for (std::size_t j = 0; j < matmul.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << matmul.output[j];
        }
        std::cout << "\n";
        std::cout << "f32_matmul_smoke_device=" << matmul.device_name << "\n";
        if (!matmul.error.empty()) std::cout << "f32_matmul_smoke_error=" << matmul.error << "\n";
        const auto dynamic_matmul = motifcl::run_vulkan_f32_m1_matmul(
            {1.0f, 2.0f, 3.0f},
            {1.0f, 2.0f,
             3.0f, 4.0f,
             5.0f, 6.0f},
            3, 2);
        std::cout << "f32_m1_matmul_success=" << (dynamic_matmul.success ? "true" : "false") << "\n";
        std::cout << "f32_m1_matmul_output=";
        for (std::size_t j = 0; j < dynamic_matmul.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << dynamic_matmul.output[j];
        }
        std::cout << "\n";
        std::cout << "f32_m1_matmul_device=" << dynamic_matmul.device_name << "\n";
        if (!dynamic_matmul.error.empty()) std::cout << "f32_m1_matmul_error=" << dynamic_matmul.error << "\n";
        const auto general_matmul = motifcl::run_vulkan_f32_matmul(
            {1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f},
            {1.0f, 2.0f,
             3.0f, 4.0f,
             5.0f, 6.0f},
            2, 3, 2);
        std::cout << "f32_general_matmul_success=" << (general_matmul.success ? "true" : "false") << "\n";
        std::cout << "f32_general_matmul_output=";
        for (std::size_t j = 0; j < general_matmul.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << general_matmul.output[j];
        }
        std::cout << "\n";
        std::cout << "f32_general_matmul_device=" << general_matmul.device_name << "\n";
        if (!general_matmul.error.empty()) std::cout << "f32_general_matmul_error=" << general_matmul.error << "\n";
    }
    if (!probe.error.empty()) std::cout << "error=" << probe.error << "\n";
    return probe.loader_found ? 0 : 77;
}
