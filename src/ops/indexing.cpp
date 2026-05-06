#include <motifcl/ops/indexing.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace motifcl {

namespace {

std::size_t element_size(DType dtype) {
    return dtype_storage_nbytes(dtype, 1);
}

std::vector<std::uint8_t> tensor_bytes(const Tensor& x) {
    std::vector<std::uint8_t> bytes(x.nbytes());
    if (!bytes.empty()) x.to_cpu(bytes.data(), bytes.size());
    return bytes;
}

struct SumAllBackwardNode : autograd::Node {
    Tensor x;
    explicit SumAllBackwardNode(Tensor x_value) : x(std::move(x_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(Tensor::full(x.backend(), x.shape(), grad_output.item(), DType::F32));
    }
};

struct MeanAllBackwardNode : autograd::Node {
    Tensor x;
    explicit MeanAllBackwardNode(Tensor x_value) : x(std::move(x_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) {
            const float scale = grad_output.item() / static_cast<float>(std::max<int64_t>(x.numel(), 1));
            x.backward(Tensor::full(x.backend(), x.shape(), scale, DType::F32));
        }
    }
};

struct SliceRowsBackwardNode : autograd::Node {
    Tensor x;
    int64_t start = 0;
    int64_t end = 0;
    SliceRowsBackwardNode(Tensor x_value, int64_t start_value, int64_t end_value)
        : x(std::move(x_value)), start(start_value), end(end_value) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (!x.requires_grad()) return;
        MCL_CHECK(x.dtype() == DType::F32, "slice_rows backward supports f32 only");
        const int64_t rows = x.shape()[0];
        const int64_t cols = x.ndim() == 1 ? 1 : x.shape()[1];
        std::vector<float> gx(static_cast<std::size_t>(rows * cols), 0.0f);
        const auto go = grad_output.to_vector<float>();
        for (int64_t r = start; r < end; ++r) {
            std::copy_n(go.data() + static_cast<std::size_t>((r - start) * cols),
                        static_cast<std::size_t>(cols),
                        gx.data() + static_cast<std::size_t>(r * cols));
        }
        x.backward(Tensor::from_cpu(x.backend(), x.shape(), DType::F32, gx.data()));
    }
};

struct MaskedFillBackwardNode : autograd::Node {
    Tensor x;
    Tensor keep;
    MaskedFillBackwardNode(Tensor x_value, Tensor keep_value) : x(std::move(x_value)), keep(std::move(keep_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(mul(grad_output, keep));
    }
};

} // namespace

Tensor sum_all(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "sum_all supports f32 only");
    const auto values = x.to_vector<float>();
    const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    auto out = Tensor::from_cpu(x.backend(), {1}, DType::F32, &sum);
    autograd::record_op("sum_all_f32", {x.id()}, {out.id()}, false);
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<SumAllBackwardNode>(x));
    }
    return out;
}

Tensor mean_all(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "mean_all supports f32 only");
    auto out = sum_all(x);
    const float mean = out.item() / static_cast<float>(std::max<int64_t>(x.numel(), 1));
    out = Tensor::from_cpu(x.backend(), {1}, DType::F32, &mean);
    autograd::record_op("mean_all_f32", {x.id()}, {out.id()}, false);
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MeanAllBackwardNode>(x));
    }
    return out;
}

Tensor slice_rows(const Tensor& x, int64_t start, int64_t end) {
    MCL_CHECK(x.ndim() == 1 || x.ndim() == 2, "slice_rows expects rank-1 or rank-2 tensor");
    if (start < 0) start += x.shape()[0];
    if (end < 0) end += x.shape()[0];
    MCL_CHECK(0 <= start && start <= end && end <= x.shape()[0], "slice_rows indices out of range");
    const int64_t cols = x.ndim() == 1 ? 1 : x.shape()[1];
    const std::size_t elem = element_size(x.dtype());
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * elem;
    const auto bytes = tensor_bytes(x);
    std::vector<std::uint8_t> out_bytes(static_cast<std::size_t>(end - start) * row_bytes);
    if (!out_bytes.empty()) {
        std::memcpy(out_bytes.data(), bytes.data() + static_cast<std::size_t>(start) * row_bytes, out_bytes.size());
    }
    const Shape out_shape = x.ndim() == 1 ? Shape({end - start}) : Shape({end - start, cols});
    auto out = Tensor::from_cpu(x.backend(), out_shape, x.dtype(), out_bytes.data());
    autograd::record_op("slice_rows", {x.id()}, {out.id()}, false);
    if (autograd::is_enabled() && x.requires_grad()) {
        MCL_CHECK(x.dtype() == DType::F32, "slice_rows autograd supports f32 only");
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<SliceRowsBackwardNode>(x, start, end));
    }
    return out;
}

Tensor dropout(const Tensor& x, float p, bool training) {
    MCL_CHECK(x.dtype() == DType::F32, "dropout supports f32 only");
    MCL_CHECK(p >= 0.0f && p < 1.0f, "dropout probability must be in [0, 1)");
    if (!training || p == 0.0f) return mul_scalar(x, 1.0f);
    const auto random = Tensor::uniform(x.backend(), x.shape(), 0.0f, 1.0f);
    const auto rv = random.to_vector<float>();
    std::vector<float> mask(rv.size(), 0.0f);
    const float keep_scale = 1.0f / (1.0f - p);
    for (std::size_t i = 0; i < rv.size(); ++i) mask[i] = rv[i] >= p ? keep_scale : 0.0f;
    auto mask_tensor = Tensor::from_cpu(x.backend(), x.shape(), DType::F32, mask.data());
    auto out = mul(x, mask_tensor);
    autograd::record_op("dropout_f32", {x.id(), mask_tensor.id()}, {out.id()}, false);
    return out;
}

Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
    MCL_CHECK(x.dtype() == DType::F32, "masked_fill supports f32 input only");
    MCL_CHECK(x.shape() == mask.shape(), "masked_fill requires same-shape mask");
    MCL_CHECK(x.backend_ptr() == mask.backend_ptr(), "masked_fill requires tensors on same backend");
    const auto xv = x.to_vector<float>();
    std::vector<float> keep(xv.size(), 1.0f);
    if (mask.dtype() == DType::F32) {
        const auto mv = mask.to_vector<float>();
        for (std::size_t i = 0; i < mv.size(); ++i) keep[i] = mv[i] != 0.0f ? 0.0f : 1.0f;
    } else if (mask.dtype() == DType::I32) {
        const auto mv = mask.to_vector<std::int32_t>();
        for (std::size_t i = 0; i < mv.size(); ++i) keep[i] = mv[i] != 0 ? 0.0f : 1.0f;
    } else if (mask.dtype() == DType::U8) {
        const auto mv = mask.to_vector<std::uint8_t>();
        for (std::size_t i = 0; i < mv.size(); ++i) keep[i] = mv[i] != 0 ? 0.0f : 1.0f;
    } else {
        MCL_CHECK(false, "masked_fill mask must be f32, i32, or u8");
    }
    std::vector<float> out_host(xv.size(), value);
    for (std::size_t i = 0; i < xv.size(); ++i) {
        if (keep[i] != 0.0f) out_host[i] = xv[i];
    }
    auto out = Tensor::from_cpu(x.backend(), x.shape(), DType::F32, out_host.data());
    auto keep_tensor = Tensor::from_cpu(x.backend(), x.shape(), DType::F32, keep.data());
    autograd::record_op("masked_fill_f32", {x.id(), mask.id()}, {out.id()}, false);
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MaskedFillBackwardNode>(x, keep_tensor));
    }
    return out;
}

} // namespace motifcl
