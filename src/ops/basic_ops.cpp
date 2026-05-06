#include <motifcl/ops/basic_ops.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/ops/reduce.hpp>

#include <algorithm>
#include <memory>
#include <numeric>

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
    auto out = Tensor::empty(a.backend(), a.shape(), DType::F32);
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

Shape broadcast_result_shape(const Shape& a, const Shape& b) {
    const std::size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int64_t> result(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const bool has_a_dim = a.dims.size() + i >= rank;
        const bool has_b_dim = b.dims.size() + i >= rank;
        const int64_t ad = has_a_dim ? a.dims[a.dims.size() + i - rank] : 1;
        const int64_t bd = has_b_dim ? b.dims[b.dims.size() + i - rank] : 1;
        MCL_CHECK(ad == bd || ad == 1 || bd == 1,
                  "broadcast shape mismatch: " + a.str() + " vs " + b.str());
        result[i] = std::max(ad, bd);
    }
    return Shape(std::move(result));
}

std::vector<int64_t> aligned_broadcast_strides(const Shape& source, const Shape& out_shape) {
    const auto source_strides = contiguous_strides(source);
    std::vector<int64_t> aligned(out_shape.dims.size(), 0);
    const std::size_t rank_delta = out_shape.dims.size() - source.dims.size();
    for (std::size_t i = 0; i < out_shape.dims.size(); ++i) {
        if (i < rank_delta) {
            aligned[i] = 0;
            continue;
        }
        const std::size_t source_dim = i - rank_delta;
        aligned[i] = source.dims[source_dim] == 1 ? 0 : source_strides[source_dim];
    }
    return aligned;
}

std::vector<float> broadcast_binary_host(const Tensor& a, const Tensor& b, const Shape& out_shape, char op) {
    MCL_CHECK(a.dtype() == DType::F32 && b.dtype() == DType::F32, "broadcast binary ops support f32 only");
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), "broadcast binary ops require tensors on same backend");
    const auto av = a.to_vector<float>();
    const auto bv = b.to_vector<float>();
    const auto out_strides = contiguous_strides(out_shape);
    const auto astrides = aligned_broadcast_strides(a.shape(), out_shape);
    const auto bstrides = aligned_broadcast_strides(b.shape(), out_shape);
    std::vector<float> out(static_cast<std::size_t>(out_shape.numel()));
    for (int64_t linear = 0; linear < out_shape.numel(); ++linear) {
        int64_t rem = linear;
        int64_t ai = 0;
        int64_t bi = 0;
        for (std::size_t d = 0; d < out_shape.dims.size(); ++d) {
            const int64_t coord = out_strides[d] ? rem / out_strides[d] : 0;
            rem = out_strides[d] ? rem % out_strides[d] : 0;
            ai += coord * astrides[d];
            bi += coord * bstrides[d];
        }
        switch (op) {
            case '+': out[static_cast<std::size_t>(linear)] = av[static_cast<std::size_t>(ai)] + bv[static_cast<std::size_t>(bi)]; break;
            case '*': out[static_cast<std::size_t>(linear)] = av[static_cast<std::size_t>(ai)] * bv[static_cast<std::size_t>(bi)]; break;
            default: MCL_CHECK(false, "unsupported broadcast op");
        }
    }
    return out;
}

Tensor reduce_broadcast_gradient(const Tensor& grad, const Shape& target_shape) {
    MCL_CHECK(grad.dtype() == DType::F32, "broadcast gradient reduction supports f32 only");
    const auto gv = grad.to_vector<float>();
    const auto& out_shape = grad.shape();
    const auto out_strides = contiguous_strides(out_shape);
    const auto target_strides = aligned_broadcast_strides(target_shape, out_shape);
    std::vector<float> reduced(static_cast<std::size_t>(target_shape.numel()), 0.0f);
    for (int64_t linear = 0; linear < out_shape.numel(); ++linear) {
        int64_t rem = linear;
        int64_t ti = 0;
        for (std::size_t d = 0; d < out_shape.dims.size(); ++d) {
            const int64_t coord = out_strides[d] ? rem / out_strides[d] : 0;
            rem = out_strides[d] ? rem % out_strides[d] : 0;
            ti += coord * target_strides[d];
        }
        reduced[static_cast<std::size_t>(ti)] += gv[static_cast<std::size_t>(linear)];
    }
    return Tensor::from_cpu(grad.backend(), target_shape, DType::F32, reduced.data());
}

Tensor elementwise_scalar(const Tensor& x, float value, const std::string& kernel_name) {
    MCL_CHECK(x.dtype() == DType::F32, kernel_name + " supports f32 only");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
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
    std::vector<Tensor> inputs() const override { return {a, b}; }
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(grad_output);
        if (b.requires_grad()) b.backward(grad_output);
    }
};

struct AddBroadcastBackward : autograd::Node {
    Tensor a, b;
    AddBroadcastBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    std::vector<Tensor> inputs() const override { return {a, b}; }
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(reduce_broadcast_gradient(grad_output, a.shape()));
        if (b.requires_grad()) b.backward(reduce_broadcast_gradient(grad_output, b.shape()));
    }
};

struct SubBackward : autograd::Node {
    Tensor a, b;
    SubBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    std::vector<Tensor> inputs() const override { return {a, b}; }
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(grad_output);
        if (b.requires_grad()) b.backward(scale(grad_output, -1.0f));
    }
};

struct MulBackward : autograd::Node {
    Tensor a, b;
    MulBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    std::vector<Tensor> inputs() const override { return {a, b}; }
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(mul(grad_output, b));
        if (b.requires_grad()) b.backward(mul(grad_output, a));
    }
};

struct MulBroadcastBackward : autograd::Node {
    Tensor a, b;
    Shape out_shape;
    MulBroadcastBackward(Tensor a, Tensor b, Shape out_shape)
        : a(std::move(a)), b(std::move(b)), out_shape(std::move(out_shape)) {}
    std::vector<Tensor> inputs() const override { return {a, b}; }
    void backward(const Tensor& grad_output) override {
        const auto gv = grad_output.to_vector<float>();
        const auto av = a.to_vector<float>();
        const auto bv = b.to_vector<float>();
        const auto out_strides = contiguous_strides(out_shape);
        const auto astrides = aligned_broadcast_strides(a.shape(), out_shape);
        const auto bstrides = aligned_broadcast_strides(b.shape(), out_shape);
        std::vector<float> ga(static_cast<std::size_t>(a.numel()), 0.0f);
        std::vector<float> gb(static_cast<std::size_t>(b.numel()), 0.0f);
        for (int64_t linear = 0; linear < out_shape.numel(); ++linear) {
            int64_t rem = linear;
            int64_t ai = 0;
            int64_t bi = 0;
            for (std::size_t d = 0; d < out_shape.dims.size(); ++d) {
                const int64_t coord = out_strides[d] ? rem / out_strides[d] : 0;
                rem = out_strides[d] ? rem % out_strides[d] : 0;
                ai += coord * astrides[d];
                bi += coord * bstrides[d];
            }
            const float go = gv[static_cast<std::size_t>(linear)];
            if (a.requires_grad()) ga[static_cast<std::size_t>(ai)] += go * bv[static_cast<std::size_t>(bi)];
            if (b.requires_grad()) gb[static_cast<std::size_t>(bi)] += go * av[static_cast<std::size_t>(ai)];
        }
        if (a.requires_grad()) a.backward(Tensor::from_cpu(grad_output.backend(), a.shape(), DType::F32, ga.data()));
        if (b.requires_grad()) b.backward(Tensor::from_cpu(grad_output.backend(), b.shape(), DType::F32, gb.data()));
    }
};

struct DivBackward : autograd::Node {
    Tensor a, b;
    DivBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    std::vector<Tensor> inputs() const override { return {a, b}; }
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
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(scale(grad_output, alpha));
    }
};

struct AddBiasRowsBackward : autograd::Node {
    Tensor x, bias;
    AddBiasRowsBackward(Tensor x, Tensor bias) : x(std::move(x)), bias(std::move(bias)) {}
    std::vector<Tensor> inputs() const override { return {x, bias}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(grad_output);
        if (bias.requires_grad()) bias.backward(sum_rows(grad_output));
    }
};

struct AddBiasGeluRowsBackward : autograd::Node {
    Tensor x, bias;
    AddBiasGeluRowsBackward(Tensor x, Tensor bias) : x(std::move(x)), bias(std::move(bias)) {}
    std::vector<Tensor> inputs() const override { return {x, bias}; }
    void backward(const Tensor& grad_output) override {
        auto preactivation = add_bias_rows(x, bias);
        auto grad_preactivation = gelu_backward_op(preactivation, grad_output);
        if (x.requires_grad()) x.backward(grad_preactivation);
        if (bias.requires_grad()) bias.backward(sum_rows(grad_preactivation));
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
    auto out = Tensor::empty(backend, shape, DType::F32);
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
    if (a.shape() != b.shape()) return add_broadcast(a, b);
    auto out = elementwise_binary(a, b, "add_f32");
    attach_binary_grad(out, a, b, std::make_shared<AddBackward>(a, b));
    return out;
}

Tensor add_broadcast(const Tensor& a, const Tensor& b) {
    if (a.shape() == b.shape()) return add(a, b);
    const auto out_shape = broadcast_result_shape(a.shape(), b.shape());
    const auto out_host = broadcast_binary_host(a, b, out_shape, '+');
    auto out = Tensor::from_cpu(a.backend(), out_shape, DType::F32, out_host.data());
    autograd::record_op("add_broadcast_f32", {a.id(), b.id()}, {out.id()}, false);
    if (autograd::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<AddBroadcastBackward>(a, b));
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto out = elementwise_binary(a, b, "sub_f32");
    attach_binary_grad(out, a, b, std::make_shared<SubBackward>(a, b));
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) return mul_broadcast(a, b);
    auto out = elementwise_binary(a, b, "mul_f32");
    attach_binary_grad(out, a, b, std::make_shared<MulBackward>(a, b));
    return out;
}

Tensor mul_broadcast(const Tensor& a, const Tensor& b) {
    if (a.shape() == b.shape()) return mul(a, b);
    const auto out_shape = broadcast_result_shape(a.shape(), b.shape());
    const auto out_host = broadcast_binary_host(a, b, out_shape, '*');
    auto out = Tensor::from_cpu(a.backend(), out_shape, DType::F32, out_host.data());
    autograd::record_op("mul_broadcast_f32", {a.id(), b.id()}, {out.id()}, false);
    if (autograd::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MulBroadcastBackward>(a, b, out_shape));
    }
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
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
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

Tensor add_bias_gelu_rows(const Tensor& x, const Tensor& bias) {
    MCL_CHECK(x.dtype() == DType::F32 && bias.dtype() == DType::F32, "add_bias_gelu_rows supports f32 only");
    MCL_CHECK(x.ndim() == 2 && bias.ndim() == 1, "add_bias_gelu_rows expects x [rows, cols] and bias [cols]");
    MCL_CHECK(x.shape()[1] == bias.shape()[0], "add_bias_gelu_rows bias dimension mismatch");
    MCL_CHECK(x.backend_ptr() == bias.backend_ptr(), "add_bias_gelu_rows requires tensors on same backend");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get("add_bias_gelu_rows_f32");
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    int n = rows * cols;
    k.set_arg(0, x.buffer());
    k.set_arg(1, bias.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("add_bias_gelu_rows_f32", {x.id(), bias.id()}, {out.id()});
    if (autograd::is_enabled() && (x.requires_grad() || bias.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<AddBiasGeluRowsBackward>(x, bias));
    }
    return out;
}

} // namespace motifcl
