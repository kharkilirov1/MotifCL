#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/microkernel.hpp>

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

bool close(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            std::fabs(actual[i] - expected[i]) > 2.0e-5f * std::max(1.0f, std::fabs(expected[i]))) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    using namespace motifcl;
    bool ok = true;
    auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << '\n';
            ok = false;
        }
    };

    expect(!close({std::numeric_limits<float>::quiet_NaN()}, {0.0f}), "NaN negative control");
    expect(!close({std::numeric_limits<float>::infinity()}, {0.0f}), "Inf negative control");
    for (const char* backend : {"opencl", "vulkan", "opencl"}) {
        set_env("MOTIFCL_ELEMENTWISE_BACKEND", backend);
        set_env("MOTIFCL_NORM_BACKEND", backend);
        set_env("MOTIFCL_ACTIVATION_BACKEND", backend);
        const auto expected = std::string(backend) == "vulkan"
            ? MicrokernelBackendKind::Vulkan : MicrokernelBackendKind::OpenCL;
        expect(selected_elementwise_backend().kind == expected, "dynamic elementwise override");
        expect(selected_norm_backend().kind == expected, "dynamic norm override");
        expect(selected_activation_backend().kind == expected, "dynamic activation override");
    }
    if (!ok) return 1;
    if (!probe_vulkan_runtime().available()) {
        std::cout << "Skipping Vulkan CPU-hotpath behavior test: no Vulkan device\n";
        return 77;
    }

    try {
        auto backend = Backend::create_vulkan();
        const Shape shape{3, 5};
        std::vector<float> xv(15), gv(15), packed_values(30);
        for (std::size_t i = 0; i < xv.size(); ++i) {
            xv[i] = (static_cast<float>(i) - 7.0f) * 0.125f;
            gv[i] = (static_cast<float>(i % 7) - 3.0f) * 0.25f;
        }
        for (std::size_t i = 0; i < packed_values.size(); ++i) {
            packed_values[i] = (static_cast<float>(i % 13) - 6.0f) * 0.125f;
        }
        auto x = Tensor::from_cpu(backend, shape, DType::F32, xv.data());
        auto g = Tensor::from_cpu(backend, shape, DType::F32, gv.data());
        auto g_alias = g.view(shape);
        auto weight = Tensor::ones(backend, {5});
        auto packed = Tensor::from_cpu(backend, {3, 10}, DType::F32, packed_values.data());
        x.set_requires_grad(true);
        weight.set_requires_grad(true);
        packed.set_requires_grad(true);

        {
            autograd::NoGradGuard guard;
            for (const auto& out : {add(x, x), sub(x, x), mul(x, x),
                                    rmsnorm(x, weight), swiglu(packed)}) {
                expect(!out.requires_grad() && !out._grad_fn(), "NoGrad must not attach backward nodes");
            }
        }
        auto twice = add(x, x);
        auto twice_view = twice.view(shape);
        auto left = add(twice, x);
        auto right = add(twice_view, x);
        auto out = add(left, right);
        expect(out.requires_grad() && out._grad_fn(), "enabled autograd must attach add backward");
        std::vector<float> expected(15);
        for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = 6.0f * gv[i];
        out.backward(g);
        expect(x.grad() && close(x.grad()->to_vector<float>(), expected), "branch/view gradient accumulation");
        expect(close(g.to_vector<float>(), gv) && close(g_alias.to_vector<float>(), gv),
               "shared upstream gradient must not be modified");
        out.backward(g_alias);
        for (auto& value : expected) value *= 2.0f;
        expect(x.grad() && close(x.grad()->to_vector<float>(), expected), "repeated backward accumulation");

        x.zero_grad();
        auto squared = mul(x, x);
        expect(squared.requires_grad() && squared._grad_fn(), "enabled autograd must attach mul backward");
        squared.backward(g);
        for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = 2.0f * xv[i] * gv[i];
        expect(x.grad() && close(x.grad()->to_vector<float>(), expected), "mul backward after lazy node creation");
        expect(rmsnorm(x, weight)._grad_fn() != nullptr, "enabled RMSNorm backward attachment");
        expect(swiglu(packed)._grad_fn() != nullptr, "enabled SwiGLU backward attachment");

        auto& runtime = backend.vulkan_runtime();
        expect(runtime.capture_begin(), "Vulkan capture must begin");
        autograd::begin_graph_capture();
        Tensor captured;
        {
            autograd::NoGradGuard guard;
            captured = add(x, x);
        }
        const auto graph = autograd::end_graph_capture();
        auto recording = runtime.capture_end();
        expect(graph.nodes().size() == 1 && graph.nodes()[0].op == "add_vulkan_f32",
               "NoGrad does not disable graph metadata capture");
        expect(recording.size() == 1, "Vulkan dispatch capture must retain the native add path");
        expect(runtime.replay(recording).success, "native add capture must replay");
        for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = 2.0f * xv[i];
        expect(close(captured.to_vector<float>(), expected), "native add replay result");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    if (ok) std::cout << "Vulkan CPU-hotpath behavior tests passed\n";
    return ok ? 0 : 1;
}
