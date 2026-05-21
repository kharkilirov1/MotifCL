#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/microkernel.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include "test_utils.hpp"

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace

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
    expect(selection.effective == MicrokernelBackendKind::Vulkan,
           "Vulkan matmul backend must select the implemented M=1 F32 descriptor");
    expect(!selection.fallback_to_opencl && selection.fallback_reason.empty(),
           "Vulkan matmul backend selection must not fallback once descriptor exists");

    const auto invalid_matmul = run_vulkan_f32_m1_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_matmul.success, "Vulkan parameterized matmul must reject malformed B size");
    expect(!invalid_matmul.error.empty(), "Vulkan parameterized matmul validation failure must explain why");
    const auto invalid_general_matmul =
        run_vulkan_f32_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2, 2);
    expect(!invalid_general_matmul.success, "Vulkan general matmul must reject malformed A size");
    expect(!invalid_general_matmul.error.empty(), "Vulkan general matmul validation failure must explain why");
    const auto invalid_softmax = run_vulkan_softmax_rows({1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_softmax.success, "Vulkan softmax rows must reject malformed input size");
    expect(!invalid_softmax.error.empty(), "Vulkan softmax rows validation failure must explain why");

    if (result.available()) {
        const bool require_vulkan_compute = std::getenv("MOTIFCL_REQUIRE_VULKAN_COMPUTE") != nullptr;
        const auto compute = run_vulkan_smoke_compute();
        if (!compute.success && !require_vulkan_compute) {
            std::cout << "Vulkan compute smoke skipped: " << compute.error << '\n';
            return ok ? 0 : 1;
        }
        expect(compute.success, "Vulkan smoke compute must succeed when strict Vulkan compute is required");
        expect(compute.error.empty(), "Vulkan smoke compute success must not carry an error");
        expect(compute.output == 42.0f, "Vulkan smoke compute must write the shader output");

        const auto fixed_matmul = run_vulkan_f32_matmul_smoke();
        expect(fixed_matmul.success, "Vulkan f32 matmul smoke must succeed when a Vulkan device is available");
        expect(fixed_matmul.error.empty(), "Vulkan f32 matmul smoke success must not carry an error");
        expect(fixed_matmul.output.size() == 4, "Vulkan f32 matmul smoke must return four output values");
        if (fixed_matmul.output.size() == 4) {
            expect(fixed_matmul.output[0] == 90.0f && fixed_matmul.output[1] == 100.0f &&
                       fixed_matmul.output[2] == 110.0f && fixed_matmul.output[3] == 120.0f,
                   "Vulkan f32 matmul smoke output mismatch");
        }

        const std::vector<float> a = {1.0f, 2.0f, 3.0f};
        const std::vector<float> b = {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
        };
        const auto dynamic_matmul = run_vulkan_f32_m1_matmul(a, b, 3, 2);
        expect(dynamic_matmul.success,
               "Vulkan parameterized f32 M=1 matmul must succeed when a Vulkan device is available");
        expect(dynamic_matmul.error.empty(),
               "Vulkan parameterized f32 M=1 matmul success must not carry an error");
        expect(dynamic_matmul.output.size() == 2,
               "Vulkan parameterized f32 M=1 matmul must return N output values");
        if (dynamic_matmul.output.size() == 2) {
            expect(dynamic_matmul.output[0] == 22.0f && dynamic_matmul.output[1] == 28.0f,
                   "Vulkan parameterized f32 M=1 matmul output mismatch");
        }

        const std::vector<float> a2 = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };
        const auto general_matmul = run_vulkan_f32_matmul(a2, b, 2, 3, 2);
        expect(general_matmul.success,
               "Vulkan general f32 matmul must succeed when a Vulkan device is available");
        expect(general_matmul.error.empty(),
               "Vulkan general f32 matmul success must not carry an error");
        expect(general_matmul.output.size() == 4,
               "Vulkan general f32 matmul must return M*N output values");
        if (general_matmul.output.size() == 4) {
            expect(general_matmul.output[0] == 22.0f && general_matmul.output[1] == 28.0f &&
                       general_matmul.output[2] == 49.0f && general_matmul.output[3] == 64.0f,
                   "Vulkan general f32 matmul output mismatch");
        }

        const std::vector<float> softmax_input = {
            1.0f, 2.0f, 3.0f,
            -1.0f, -1.0f, -1.0f,
        };
        const auto softmax = run_vulkan_softmax_rows(softmax_input, 2, 3);
        expect(softmax.success,
               "Vulkan softmax rows must succeed when a Vulkan device is available");
        expect(softmax.error.empty(),
               "Vulkan softmax rows success must not carry an error");
        expect(softmax.output.size() == softmax_input.size(),
               "Vulkan softmax rows must return rows*cols output values");
        if (softmax.output.size() == softmax_input.size()) {
            const float denom = std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f;
            expect(std::fabs(softmax.output[0] - std::exp(1.0f - 3.0f) / denom) <= 1e-5f &&
                       std::fabs(softmax.output[1] - std::exp(2.0f - 3.0f) / denom) <= 1e-5f &&
                       std::fabs(softmax.output[2] - 1.0f / denom) <= 1e-5f,
                   "Vulkan softmax rows first row output mismatch");
            expect(std::fabs(softmax.output[3] - (1.0f / 3.0f)) <= 1e-5f &&
                       std::fabs(softmax.output[4] - (1.0f / 3.0f)) <= 1e-5f &&
                       std::fabs(softmax.output[5] - (1.0f / 3.0f)) <= 1e-5f,
                   "Vulkan softmax rows uniform row output mismatch");
        }

        set_env("MOTIFCL_MATMUL_BACKEND", "vulkan");
        try {
            auto backend = Backend::create_opencl();
            auto A = Tensor::from_cpu(backend, {1, 3}, DType::F32, a.data());
            auto B = Tensor::from_cpu(backend, {3, 2}, DType::F32, b.data());
            autograd::begin_graph_capture();
            auto C = matmul(A, B);
            auto graph = autograd::end_graph_capture();
            const auto c = C.to_vector<float>();
            expect(c.size() == 2, "Vulkan matmul dispatch must return N output values");
            if (c.size() == 2) {
                expect(std::fabs(c[0] - 22.0f) <= 1e-5f && std::fabs(c[1] - 28.0f) <= 1e-5f,
                       "Vulkan matmul dispatch output mismatch");
            }
            expect(!graph.empty() && graph.nodes()[0].op == "matmul_vulkan_f32_m1",
                   "Vulkan matmul dispatch must record the Vulkan M=1 op");

            auto A2 = Tensor::from_cpu(backend, {2, 3}, DType::F32, a2.data());
            autograd::begin_graph_capture();
            auto C2 = matmul(A2, B);
            auto graph2 = autograd::end_graph_capture();
            const auto c2 = C2.to_vector<float>();
            expect(c2.size() == 4, "Vulkan general matmul dispatch must return M*N output values");
            if (c2.size() == 4) {
                expect(std::fabs(c2[0] - 22.0f) <= 1e-5f && std::fabs(c2[1] - 28.0f) <= 1e-5f &&
                           std::fabs(c2[2] - 49.0f) <= 1e-5f && std::fabs(c2[3] - 64.0f) <= 1e-5f,
                       "Vulkan general matmul dispatch output mismatch");
            }
            expect(!graph2.empty() && graph2.nodes()[0].op == "matmul_vulkan_f32",
                   "Vulkan general matmul dispatch must record the Vulkan F32 op");

            set_env("MOTIFCL_ATTENTION_BACKEND", "vulkan");
            auto SX = Tensor::from_cpu(backend, {2, 3}, DType::F32, softmax_input.data());
            autograd::begin_graph_capture();
            auto SY = softmax_rows(SX);
            auto softmax_graph = autograd::end_graph_capture();
            const auto sy = SY.to_vector<float>();
            expect(sy.size() == softmax_input.size(), "Vulkan softmax dispatch must return rows*cols values");
            if (sy.size() == softmax_input.size()) {
                expect(std::fabs(sy[0] - softmax.output[0]) <= 1e-5f &&
                           std::fabs(sy[1] - softmax.output[1]) <= 1e-5f &&
                           std::fabs(sy[2] - softmax.output[2]) <= 1e-5f &&
                           std::fabs(sy[3] - softmax.output[3]) <= 1e-5f,
                       "Vulkan softmax dispatch output mismatch");
            }
            expect(!softmax_graph.empty() && softmax_graph.nodes()[0].op == "softmax_rows_vulkan_f32",
                   "Vulkan softmax dispatch must record the Vulkan softmax op");
            set_env("MOTIFCL_ATTENTION_BACKEND", "opencl");
        } catch (const std::exception& e) {
            if (motifcl_test::is_opencl_unavailable(e)) {
                std::cout << "Vulkan matmul dispatch smoke skipped because OpenCL tensors are unavailable: "
                          << e.what() << '\n';
            } else {
                std::cerr << "Vulkan matmul dispatch smoke failed: " << e.what() << '\n';
                ok = false;
            }
        }
        set_env("MOTIFCL_MATMUL_BACKEND", "opencl");
        set_env("MOTIFCL_ATTENTION_BACKEND", "opencl");
    }

    return ok ? 0 : 1;
}
