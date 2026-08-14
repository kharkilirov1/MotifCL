#include <motifcl/ops/activation.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/runtime/microkernel.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace motifcl {

namespace {
constexpr std::size_t kLocal = 256;
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }

bool strict_vulkan_activation_required() {
    const auto enabled = [](const char* value) {
        if (value == nullptr) return false;
        const std::string text(value);
        return text == "1" || text == "true" || text == "TRUE" ||
               text == "on" || text == "ON" || text == "yes" || text == "YES";
    };
    return enabled(std::getenv("MOTIFCL_REQUIRE_VULKAN_COMPUTE")) ||
           enabled(std::getenv("MOTIFCL_REQUIRE_VULKAN_ACTIVATION"));
}

Tensor unary(const Tensor& x, const std::string& kernel_name) {
    MCL_CHECK(x.dtype() == DType::F32, kernel_name + " supports f32 only");
    // Only gelu has a Vulkan device path; relu/silu/exp/sqrt/rsqrt still go
    // through the OpenCL kernel cache. Refuse Vulkan so the failure is loud.
    MCL_CHECK(!x.backend().is_vulkan(),
              std::string(kernel_name) + " is not Vulkan-native yet (only gelu/swiglu have Vulkan "
              "device paths; use OpenCL backend)");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get(kernel_name);
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op(kernel_name, {x.id()}, {out.id()});
    return out;
}

Tensor unary_backward_kernel(const Tensor& x, const Tensor& grad_out, const std::string& kernel_name) {
    MCL_CHECK(x.dtype() == DType::F32 && grad_out.dtype() == DType::F32, kernel_name + " supports f32 only");
    MCL_CHECK(x.shape() == grad_out.shape(), kernel_name + " shape mismatch");
    // gelu_backward has its own Vulkan device path; relu_backward (and any
    // future unary_backward) does not. Refuse Vulkan so the failure is loud.
    MCL_CHECK(!x.backend().is_vulkan(),
              std::string(kernel_name) + " is not Vulkan-native yet (only gelu_backward has a Vulkan "
              "device path; use OpenCL backend)");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get(kernel_name);
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, grad_out.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op(kernel_name, {x.id(), grad_out.id()}, {out.id()});
    return out;
}

struct ReluBackwardNode : autograd::Node {
    Tensor x;
    explicit ReluBackwardNode(Tensor x) : x(std::move(x)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(relu_backward_op(x, grad_output));
    }
};

struct GeluBackwardNode : autograd::Node {
    Tensor x;
    explicit GeluBackwardNode(Tensor x) : x(std::move(x)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(gelu_backward_op(x, grad_output));
    }
};

struct SiluBackwardNode : autograd::Node {
    Tensor x;
    explicit SiluBackwardNode(Tensor x_value) : x(std::move(x_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(silu_backward_op(x, grad_output));
    }
};

struct SigmoidBackwardNode : autograd::Node {
    Tensor y;
    Tensor x;
    SigmoidBackwardNode(Tensor y_value, Tensor x_value) : y(std::move(y_value)), x(std::move(x_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(sigmoid_backward_op(y, grad_output));
    }
};

struct SwiGLUBackwardNode : autograd::Node {
    Tensor packed;
    explicit SwiGLUBackwardNode(Tensor packed_value) : packed(std::move(packed_value)) {}
    std::vector<Tensor> inputs() const override { return {packed}; }
    void backward(const Tensor& grad_output) override {
        if (packed.requires_grad()) packed.backward(swiglu_backward_op(packed, grad_output));
    }
};

bool vulkan_swiglu_supported(const Tensor& packed) {
    const bool base = packed.dtype() == DType::F32 &&
                      packed.ndim() == 2 &&
                      packed.shape()[0] > 0 &&
                      packed.shape()[1] > 0 &&
                      (packed.shape()[1] % 2) == 0;
    if (!base) return false;
    if (packed.backend().is_vulkan()) return true;
    return packed.shape()[0] <= 4096 && (packed.shape()[1] / 2) <= 4096 && !packed.requires_grad();
}

Tensor swiglu_vulkan_f32(const Tensor& packed, const VulkanF32TensorResult& result) {
    MCL_CHECK(result.success, std::string("vulkan swiglu f32 failed: ") + result.error);
    const auto hidden = packed.shape()[1] / 2;
    const auto expected = static_cast<std::size_t>(packed.shape()[0] * hidden);
    MCL_CHECK(result.output.size() == expected, "vulkan swiglu f32 returned unexpected output size");
    auto out = Tensor::from_cpu(packed.backend(), {packed.shape()[0], hidden}, DType::F32, result.output.data());
    autograd::record_op("swiglu_vulkan_f32", {packed.id()}, {out.id()});
    return out;
}

Tensor swiglu_vulkan_f32_device(const Tensor& packed) {
    const auto hidden = packed.shape()[1] / 2;
    auto out = Tensor::empty(packed.backend(), {packed.shape()[0], hidden}, DType::F32);
    const auto result = run_vulkan_swiglu(packed.backend().vulkan_runtime(),
                                          packed.storage().vulkan_buffer,
                                          out.storage().vulkan_buffer,
                                          static_cast<std::size_t>(packed.shape()[0]),
                                          static_cast<std::size_t>(hidden));
    MCL_CHECK(result.success, std::string("vulkan swiglu f32 failed: ") + result.error);
    autograd::record_op("swiglu_vulkan_f32", {packed.id()}, {out.id()});
    return out;
}

} // namespace

Tensor relu(const Tensor& x) {
    auto out = unary(x, "relu_f32");
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<ReluBackwardNode>(x));
    }
    return out;
}

Tensor relu_backward_op(const Tensor& x, const Tensor& grad_out) {
    return unary_backward_kernel(x, grad_out, "relu_backward_f32");
}

Tensor gelu(const Tensor& x) {
    Tensor out;
    if (x.backend().is_vulkan()) {
        MCL_CHECK(x.dtype() == DType::F32, "gelu supports f32 only");
        out = Tensor::empty(x.backend(), x.shape(), DType::F32);
        const auto result = run_vulkan_gelu(x.backend().vulkan_runtime(),
                                            x.storage().vulkan_buffer,
                                            out.storage().vulkan_buffer,
                                            static_cast<std::size_t>(x.numel()));
        MCL_CHECK(result.success, std::string("vulkan gelu failed: ") + result.error);
        autograd::record_op("gelu_vulkan_f32", {x.id()}, {out.id()});
    } else {
        out = unary(x, "gelu_f32");
    }
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<GeluBackwardNode>(x));
    }
    return out;
}

Tensor gelu_backward_op(const Tensor& x, const Tensor& grad_out) {
    if (x.backend().is_vulkan()) {
        MCL_CHECK(x.dtype() == DType::F32 && grad_out.dtype() == DType::F32, "gelu_backward supports f32 only");
        MCL_CHECK(x.shape() == grad_out.shape(), "gelu_backward shape mismatch");
        auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
        const auto result = run_vulkan_gelu_backward(x.backend().vulkan_runtime(),
                                                     x.storage().vulkan_buffer,
                                                     grad_out.storage().vulkan_buffer,
                                                     out.storage().vulkan_buffer,
                                                     static_cast<std::size_t>(x.numel()));
        MCL_CHECK(result.success, std::string("vulkan gelu backward failed: ") + result.error);
        autograd::record_op("gelu_backward_vulkan_f32", {x.id(), grad_out.id()}, {out.id()});
        return out;
    }
    return unary_backward_kernel(x, grad_out, "gelu_backward_f32");
}

Tensor silu(const Tensor& x) {
    Tensor out;
    if (x.backend().is_vulkan()) {
        MCL_CHECK(x.dtype() == DType::F32, "silu supports f32 only");
        out = Tensor::empty(x.backend(), x.shape(), DType::F32);
        const auto result = run_vulkan_silu(x.backend().vulkan_runtime(),
                                            x.storage().vulkan_buffer,
                                            out.storage().vulkan_buffer,
                                            static_cast<std::size_t>(x.numel()));
        MCL_CHECK(result.success, std::string("vulkan silu failed: ") + result.error);
        autograd::record_op("silu_vulkan_f32", {x.id()}, {out.id()});
    } else {
        out = unary(x, "silu_f32");
    }
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<SiluBackwardNode>(x));
    }
    return out;
}

Tensor silu_backward_op(const Tensor& x, const Tensor& grad_out) {
    if (x.backend().is_vulkan()) {
        MCL_CHECK(x.dtype() == DType::F32 && grad_out.dtype() == DType::F32, "silu_backward supports f32 only");
        MCL_CHECK(x.shape() == grad_out.shape(), "silu_backward shape mismatch");
        auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
        const auto result = run_vulkan_silu_backward(x.backend().vulkan_runtime(),
                                                     x.storage().vulkan_buffer,
                                                     grad_out.storage().vulkan_buffer,
                                                     out.storage().vulkan_buffer,
                                                     static_cast<std::size_t>(x.numel()));
        MCL_CHECK(result.success, std::string("vulkan silu backward failed: ") + result.error);
        autograd::record_op("silu_backward_vulkan_f32", {x.id(), grad_out.id()}, {out.id()});
        return out;
    }
    // Reuse an explicit host/OpenCL expression through primitive ops so the
    // formula is identical to the Vulkan kernel and remains differentiability-independent.
    const auto xv = x.to_vector<float>();
    const auto gv = grad_out.to_vector<float>();
    std::vector<float> dx(xv.size());
    for (std::size_t i = 0; i < xv.size(); ++i) {
        const float s = 1.0f / (1.0f + std::exp(-xv[i]));
        dx[i] = gv[i] * s * (1.0f + xv[i] * (1.0f - s));
    }
    return Tensor::from_cpu(x.backend(), x.shape(), DType::F32, dx.data());
}

Tensor sigmoid(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "sigmoid supports f32 only");
    Tensor out;
    if (x.backend().is_vulkan()) {
        out = Tensor::empty(x.backend(), x.shape(), DType::F32);
        const auto result = run_vulkan_sigmoid(x.backend().vulkan_runtime(),
                                               x.storage().vulkan_buffer,
                                               out.storage().vulkan_buffer,
                                               static_cast<std::size_t>(x.numel()));
        MCL_CHECK(result.success, std::string("vulkan sigmoid failed: ") + result.error);
        autograd::record_op("sigmoid_vulkan_f32", {x.id()}, {out.id()});
    } else {
        const auto xv = x.to_vector<float>();
        std::vector<float> y(xv.size());
        for (std::size_t i = 0; i < xv.size(); ++i) y[i] = 1.0f / (1.0f + std::exp(-xv[i]));
        out = Tensor::from_cpu(x.backend(), x.shape(), DType::F32, y.data());
        autograd::record_op("sigmoid_host_f32", {x.id()}, {out.id()}, false);
    }
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<SigmoidBackwardNode>(out, x));
    }
    return out;
}

Tensor sigmoid_backward_op(const Tensor& y, const Tensor& grad_out) {
    MCL_CHECK(y.dtype() == DType::F32 && grad_out.dtype() == DType::F32, "sigmoid_backward supports f32 only");
    MCL_CHECK(y.shape() == grad_out.shape(), "sigmoid_backward shape mismatch");
    if (y.backend().is_vulkan()) {
        auto out = Tensor::empty(y.backend(), y.shape(), DType::F32);
        const auto result = run_vulkan_sigmoid_backward(y.backend().vulkan_runtime(),
                                                        y.storage().vulkan_buffer,
                                                        grad_out.storage().vulkan_buffer,
                                                        out.storage().vulkan_buffer,
                                                        static_cast<std::size_t>(y.numel()));
        MCL_CHECK(result.success, std::string("vulkan sigmoid backward failed: ") + result.error);
        autograd::record_op("sigmoid_backward_vulkan_f32", {y.id(), grad_out.id()}, {out.id()});
        return out;
    }
    const auto yv = y.to_vector<float>();
    const auto gv = grad_out.to_vector<float>();
    std::vector<float> dx(yv.size());
    for (std::size_t i = 0; i < yv.size(); ++i) dx[i] = gv[i] * yv[i] * (1.0f - yv[i]);
    return Tensor::from_cpu(y.backend(), y.shape(), DType::F32, dx.data());
}

Tensor swiglu(const Tensor& packed) {
    MCL_CHECK(packed.dtype() == DType::F32, "swiglu supports f32 only");
    MCL_CHECK(packed.ndim() == 2 && packed.shape()[1] % 2 == 0, "swiglu expects [rows, 2*hidden]");
    const auto selected_backend = selected_activation_backend();
    if ((packed.backend().is_vulkan() || selected_backend.kind == MicrokernelBackendKind::Vulkan) &&
        vulkan_swiglu_supported(packed)) {
        if (packed.backend().is_vulkan()) {
            auto out = swiglu_vulkan_f32_device(packed);
            if (autograd::is_enabled() && packed.requires_grad()) {
                out.set_requires_grad(true);
                out._set_grad_fn(std::make_shared<SwiGLUBackwardNode>(packed));
            }
            return out;
        }
        const auto rows = static_cast<std::size_t>(packed.shape()[0]);
        const auto hidden = static_cast<std::size_t>(packed.shape()[1] / 2);
        const auto packed_host = packed.to_vector<float>();
        const auto result = run_vulkan_swiglu(packed_host, rows, hidden);
        if (result.success) return swiglu_vulkan_f32(packed, result);
        MCL_CHECK(!strict_vulkan_activation_required(),
                  std::string("vulkan swiglu f32 failed: ") + result.error);
    }
    MCL_CHECK(!packed.backend().is_vulkan(), "vulkan backend does not support this swiglu shape");
    const int rows = static_cast<int>(packed.shape()[0]);
    const int hidden = static_cast<int>(packed.shape()[1] / 2);
    auto out = Tensor::empty(packed.backend(), {packed.shape()[0], hidden}, DType::F32);
    auto k = packed.backend().kernels.get("swiglu_f32");
    k.set_arg(0, packed.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, hidden);
    k.launch1d(round_up(static_cast<std::size_t>(rows * hidden), kLocal), kLocal);
    autograd::record_op("swiglu_f32", {packed.id()}, {out.id()});
    if (autograd::is_enabled() && packed.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<SwiGLUBackwardNode>(packed));
    }
    return out;
}

Tensor swiglu_backward_op(const Tensor& packed, const Tensor& grad_out) {
    MCL_CHECK(packed.dtype() == DType::F32 && grad_out.dtype() == DType::F32, "swiglu_backward supports f32 only");
    MCL_CHECK(packed.ndim() == 2 && packed.shape()[1] % 2 == 0, "swiglu_backward expects packed [rows, 2*hidden]");
    MCL_CHECK(grad_out.shape() == Shape({packed.shape()[0], packed.shape()[1] / 2}), "swiglu_backward grad_out shape mismatch");
    if (packed.backend().is_vulkan()) {
        auto out = Tensor::empty(packed.backend(), packed.shape(), DType::F32);
        const auto result = run_vulkan_swiglu_backward(packed.backend().vulkan_runtime(),
                                                       packed.storage().vulkan_buffer,
                                                       grad_out.storage().vulkan_buffer,
                                                       out.storage().vulkan_buffer,
                                                       static_cast<std::size_t>(packed.shape()[0]),
                                                       static_cast<std::size_t>(packed.shape()[1] / 2));
        MCL_CHECK(result.success, std::string("vulkan swiglu backward failed: ") + result.error);
        autograd::record_op("swiglu_backward_vulkan_f32", {packed.id(), grad_out.id()}, {out.id()});
        return out;
    }
    const int rows = static_cast<int>(packed.shape()[0]);
    const int hidden = static_cast<int>(packed.shape()[1] / 2);
    auto out = Tensor::empty(packed.backend(), packed.shape(), DType::F32);
    auto k = packed.backend().kernels.get("swiglu_backward_f32");
    k.set_arg(0, packed.buffer());
    k.set_arg(1, grad_out.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, hidden);
    k.launch1d(round_up(static_cast<std::size_t>(packed.numel()), kLocal), kLocal);
    autograd::record_op("swiglu_backward_f32", {packed.id(), grad_out.id()}, {out.id()});
    return out;
}

Tensor exp(const Tensor& x) { return unary(x, "exp_f32"); }
Tensor sqrt(const Tensor& x) { return unary(x, "sqrt_f32"); }
Tensor rsqrt(const Tensor& x) { return unary(x, "rsqrt_f32"); }

} // namespace motifcl
