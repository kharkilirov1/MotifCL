#include <motifcl/ops/activation.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

#include <memory>

namespace motifcl {

namespace {
constexpr std::size_t kLocal = 256;
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }

Tensor unary(const Tensor& x, const std::string& kernel_name) {
    MCL_CHECK(x.dtype() == DType::F32, kernel_name + " supports f32 only");
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

struct SwiGLUBackwardNode : autograd::Node {
    Tensor packed;
    explicit SwiGLUBackwardNode(Tensor packed_value) : packed(std::move(packed_value)) {}
    std::vector<Tensor> inputs() const override { return {packed}; }
    void backward(const Tensor& grad_output) override {
        if (packed.requires_grad()) packed.backward(swiglu_backward_op(packed, grad_output));
    }
};

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
    auto out = unary(x, "gelu_f32");
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<GeluBackwardNode>(x));
    }
    return out;
}

Tensor gelu_backward_op(const Tensor& x, const Tensor& grad_out) {
    return unary_backward_kernel(x, grad_out, "gelu_backward_f32");
}

Tensor silu(const Tensor& x) { return unary(x, "silu_f32"); }

Tensor swiglu(const Tensor& packed) {
    MCL_CHECK(packed.dtype() == DType::F32, "swiglu supports f32 only");
    MCL_CHECK(packed.ndim() == 2 && packed.shape()[1] % 2 == 0, "swiglu expects [rows, 2*hidden]");
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
