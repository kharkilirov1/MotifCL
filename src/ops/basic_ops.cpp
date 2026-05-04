#include <motifcl/ops/basic_ops.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/ops/reduce.hpp>

#include <memory>

namespace motifcl {

namespace {

constexpr std::size_t kLocal1D = 256;

std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

void require_f32_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    MCL_CHECK(a.dtype() == DType::F32 && b.dtype() == DType::F32, std::string(op) + " supports f32 only");
    MCL_CHECK(a.shape() == b.shape(), std::string(op) + " shape mismatch: " + a.shape().str() + " vs " + b.shape().str());
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), std::string(op) + " requires tensors on same backend");
}

Tensor elementwise_binary(const Tensor& a, const Tensor& b, const std::string& kernel_name) {
    require_f32_same_shape(a, b, kernel_name.c_str());
    auto out = Tensor::zeros(a.backend(), a.shape(), DType::F32);
    auto k = a.backend().kernels.get(kernel_name);
    int n = static_cast<int>(a.numel());
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op(kernel_name, {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor elementwise_scalar(const Tensor& x, float value, const std::string& kernel_name) {
    MCL_CHECK(x.dtype() == DType::F32, kernel_name + " supports f32 only");
    auto out = Tensor::zeros(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get(kernel_name);
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, value);
    k.set_arg(2, out.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op(kernel_name, {x.id()}, {out.id()});
    return out;
}

struct AddBackward : autograd::Node {
    Tensor a, b;
    AddBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(grad_output);
        if (b.requires_grad()) b.backward(grad_output);
    }
};

struct SubBackward : autograd::Node {
    Tensor a, b;
    SubBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(grad_output);
        if (b.requires_grad()) b.backward(scale(grad_output, -1.0f));
    }
};

struct MulBackward : autograd::Node {
    Tensor a, b;
    MulBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(mul(grad_output, b));
        if (b.requires_grad()) b.backward(mul(grad_output, a));
    }
};

struct DivBackward : autograd::Node {
    Tensor a, b;
    DivBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(div(grad_output, b));
        if (b.requires_grad()) {
            auto b2 = mul(b, b);
            b.backward(scale(div(mul(grad_output, a), b2), -1.0f));
        }
    }
};

struct ScalarBackward : autograd::Node {
    Tensor x;
    float alpha = 1.0f;
    ScalarBackward(Tensor x, float alpha) : x(std::move(x)), alpha(alpha) {}
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(scale(grad_output, alpha));
    }
};

struct AddBiasRowsBackward : autograd::Node {
    Tensor x, bias;
    AddBiasRowsBackward(Tensor x, Tensor bias) : x(std::move(x)), bias(std::move(bias)) {}
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(grad_output);
        if (bias.requires_grad()) bias.backward(sum_rows(grad_output));
    }
};

void attach_binary_grad(Tensor& out, const Tensor& a, const Tensor& b, std::shared_ptr<autograd::Node> node) {
    if (autograd::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::move(node));
    }
}

} // namespace

Tensor fill(Backend& backend, const Shape& shape, float value) {
    auto out = Tensor::zeros(backend, shape, DType::F32);
    auto k = backend.kernels.get("fill_f32");
    int n = static_cast<int>(out.numel());
    k.set_arg(0, out.buffer());
    k.set_arg(1, value);
    k.set_arg(2, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("fill_f32", {}, {out.id()});
    return out;
}

Tensor zeros_like(const Tensor& x) { return Tensor::zeros(x.backend(), x.shape(), x.dtype()); }
Tensor ones_like(const Tensor& x) { return Tensor::ones(x.backend(), x.shape(), x.dtype()); }

Tensor add(const Tensor& a, const Tensor& b) {
    auto out = elementwise_binary(a, b, "add_f32");
    attach_binary_grad(out, a, b, std::make_shared<AddBackward>(a, b));
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto out = elementwise_binary(a, b, "sub_f32");
    attach_binary_grad(out, a, b, std::make_shared<SubBackward>(a, b));
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    auto out = elementwise_binary(a, b, "mul_f32");
    attach_binary_grad(out, a, b, std::make_shared<MulBackward>(a, b));
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    auto out = elementwise_binary(a, b, "div_f32");
    attach_binary_grad(out, a, b, std::make_shared<DivBackward>(a, b));
    return out;
}

Tensor add_scalar(const Tensor& x, float value) {
    auto out = elementwise_scalar(x, value, "add_scalar_f32");
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<ScalarBackward>(x, 1.0f));
    }
    return out;
}

Tensor mul_scalar(const Tensor& x, float value) { return scale(x, value); }

Tensor scale(const Tensor& x, float alpha) {
    auto out = elementwise_scalar(x, alpha, "mul_scalar_f32");
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<ScalarBackward>(x, alpha));
    }
    return out;
}

void scale_inplace(Tensor& x, float alpha) {
    MCL_CHECK(x.dtype() == DType::F32, "scale_inplace supports f32 only");
    auto k = x.backend().kernels.get("scale_f32");
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, alpha);
    k.set_arg(2, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("scale_inplace_f32", {x.id()}, {x.id()});
}

void add_inplace(Tensor& dst, const Tensor& src) {
    require_f32_same_shape(dst, src, "add_inplace");
    auto k = dst.backend().kernels.get("add_inplace_f32");
    int n = static_cast<int>(dst.numel());
    k.set_arg(0, dst.buffer());
    k.set_arg(1, src.buffer());
    k.set_arg(2, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("add_inplace_f32", {dst.id(), src.id()}, {dst.id()});
}

Tensor add_bias_rows(const Tensor& x, const Tensor& bias) {
    MCL_CHECK(x.dtype() == DType::F32 && bias.dtype() == DType::F32, "add_bias_rows supports f32 only");
    MCL_CHECK(x.ndim() == 2 && bias.ndim() == 1, "add_bias_rows expects x [rows, cols] and bias [cols]");
    MCL_CHECK(x.shape()[1] == bias.shape()[0], "bias dimension mismatch");
    MCL_CHECK(x.backend_ptr() == bias.backend_ptr(), "add_bias_rows requires tensors on same backend");
    auto out = Tensor::zeros(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get("add_bias_rows_f32");
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    int n = rows * cols;
    k.set_arg(0, x.buffer());
    k.set_arg(1, bias.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("add_bias_rows_f32", {x.id(), bias.id()}, {out.id()});
    if (autograd::is_enabled() && (x.requires_grad() || bias.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<AddBiasRowsBackward>(x, bias));
    }
    return out;
}

} // namespace motifcl
