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
        const auto softmax = motifcl::run_vulkan_softmax_rows(
            {1.0f, 2.0f, 3.0f,
             -1.0f, -1.0f, -1.0f},
            2, 3);
        std::cout << "softmax_rows_success=" << (softmax.success ? "true" : "false") << "\n";
        std::cout << "softmax_rows_output=";
        for (std::size_t j = 0; j < softmax.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << softmax.output[j];
        }
        std::cout << "\n";
        std::cout << "softmax_rows_device=" << softmax.device_name << "\n";
        if (!softmax.error.empty()) std::cout << "softmax_rows_error=" << softmax.error << "\n";
        const auto rmsnorm = motifcl::run_vulkan_rmsnorm(
            {1.0f, 2.0f, 3.0f, 4.0f,
             -2.0f, 0.0f, 2.0f, 4.0f},
            {1.0f, 0.5f, 1.5f, 2.0f},
            2, 4, 1.0e-6f);
        std::cout << "rmsnorm_success=" << (rmsnorm.success ? "true" : "false") << "\n";
        std::cout << "rmsnorm_output=";
        for (std::size_t j = 0; j < rmsnorm.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << rmsnorm.output[j];
        }
        std::cout << "\n";
        std::cout << "rmsnorm_device=" << rmsnorm.device_name << "\n";
        if (!rmsnorm.error.empty()) std::cout << "rmsnorm_error=" << rmsnorm.error << "\n";
        const auto swiglu = motifcl::run_vulkan_swiglu(
            {1.0f, -2.0f, 0.5f, 4.0f,
             -1.5f, 3.0f, -2.0f, 0.25f},
            2, 2);
        std::cout << "swiglu_success=" << (swiglu.success ? "true" : "false") << "\n";
        std::cout << "swiglu_output=";
        for (std::size_t j = 0; j < swiglu.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << swiglu.output[j];
        }
        std::cout << "\n";
        std::cout << "swiglu_device=" << swiglu.device_name << "\n";
        if (!swiglu.error.empty()) std::cout << "swiglu_error=" << swiglu.error << "\n";
        const auto add = motifcl::run_vulkan_add(
            {1.0f, -2.0f, 3.5f, 4.0f},
            {4.0f, 5.0f, -0.5f, -1.0f});
        std::cout << "add_success=" << (add.success ? "true" : "false") << "\n";
        std::cout << "add_output=";
        for (std::size_t j = 0; j < add.output.size(); ++j) {
            if (j != 0) std::cout << ",";
            std::cout << add.output[j];
        }
        std::cout << "\n";
        std::cout << "add_device=" << add.device_name << "\n";
        if (!add.error.empty()) std::cout << "add_error=" << add.error << "\n";
    }
    if (!probe.error.empty()) std::cout << "error=" << probe.error << "\n";
    return probe.loader_found ? 0 : 77;
}
