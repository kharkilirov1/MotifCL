#include <motifcl/ops/indexing.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/runtime/backend.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <atomic>
#include <vector>

namespace motifcl {

namespace {
constexpr std::size_t kLocal1D = 256;
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }

std::atomic<std::uint32_t> g_dropout_counter{0x12345678u};

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
    Tensor mask;
    MaskedFillBackwardNode(Tensor x_value, Tensor mask_value) : x(std::move(x_value)), mask(std::move(mask_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(masked_fill(grad_output, mask, 0.0f));
    }
};

struct DropoutBackwardNode : autograd::Node {
    Tensor x;
    Tensor mask;
    DropoutBackwardNode(Tensor x_value, Tensor mask_value) : x(std::move(x_value)), mask(std::move(mask_value)) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override {
        if (x.requires_grad()) x.backward(mul(grad_output, mask));
    }
};

} // namespace

void dropout_manual_seed(std::uint32_t seed) {
    g_dropout_counter.store(seed ^ 0x12345678u, std::memory_order_relaxed);
}

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

Tensor gather_last_token_logits(const Tensor& logits,
                                const Tensor& positions,
                                int64_t batch_size,
                                int64_t seq_len,
                                int64_t vocab_size) {
    MCL_CHECK(logits.dtype() == DType::F32, "gather_last_token_logits supports f32 logits only");
    MCL_CHECK(positions.dtype() == DType::I32, "gather_last_token_logits positions must be i32");
    MCL_CHECK(positions.ndim() == 1 && positions.shape()[0] == batch_size,
              "gather_last_token_logits positions must be [batch]");
    MCL_CHECK(batch_size > 0 && seq_len > 0 && vocab_size > 0,
              "gather_last_token_logits invalid batch/seq/vocab dimensions");
    MCL_CHECK(logits.numel() == batch_size * seq_len * vocab_size,
              "gather_last_token_logits logits size mismatch");
    MCL_CHECK(logits.backend_ptr() == positions.backend_ptr(),
              "gather_last_token_logits tensors must share backend");
    auto out = Tensor::empty(logits.backend(), {batch_size, vocab_size}, DType::F32);
    auto k = logits.backend().kernels.get("gather_last_token_logits_f32");
    const int n = static_cast<int>(batch_size * vocab_size);
    k.set_arg(0, logits.buffer());
    k.set_arg(1, positions.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, static_cast<int>(batch_size));
    k.set_arg(4, static_cast<int>(seq_len));
    k.set_arg(5, static_cast<int>(vocab_size));
    k.set_arg(6, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("gather_last_token_logits_f32", {logits.id(), positions.id()}, {out.id()});
    return out;
}

Tensor dropout(const Tensor& x, float p, bool training) {
    MCL_CHECK(x.dtype() == DType::F32, "dropout supports f32 only");
    MCL_CHECK(p >= 0.0f && p < 1.0f, "dropout probability must be in [0, 1)");
    if (!training || p == 0.0f) return mul_scalar(x, 1.0f);
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto mask_tensor = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const float keep_scale = 1.0f / (1.0f - p);
    const std::uint32_t seed = g_dropout_counter.fetch_add(0x9e3779b9u, std::memory_order_relaxed) ^
                               static_cast<std::uint32_t>(x.id() * 0x85ebca6bu);
    auto k = x.backend().kernels.get("dropout_f32");
    const int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, mask_tensor.buffer());
    k.set_arg(3, n);
    k.set_arg(4, p);
    k.set_arg(5, keep_scale);
    k.set_arg(6, seed);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op("dropout_f32", {x.id()}, {out.id(), mask_tensor.id()});
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<DropoutBackwardNode>(x, mask_tensor));
    }
    return out;
}

Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
    MCL_CHECK(x.dtype() == DType::F32, "masked_fill supports f32 input only");
    MCL_CHECK(x.shape() == mask.shape(), "masked_fill requires same-shape mask");
    MCL_CHECK(x.backend_ptr() == mask.backend_ptr(), "masked_fill requires tensors on same backend");
    std::string kernel_name;
    if (mask.dtype() == DType::F32) kernel_name = "masked_fill_f32_mask_f32";
    else if (mask.dtype() == DType::I32) kernel_name = "masked_fill_f32_mask_i32";
    else if (mask.dtype() == DType::U8) kernel_name = "masked_fill_f32_mask_u8";
    else MCL_CHECK(false, "masked_fill mask must be f32, i32, or u8");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get(kernel_name);
    const int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, mask.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, value);
    k.set_arg(4, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal1D), kLocal1D);
    autograd::record_op(kernel_name, {x.id(), mask.id()}, {out.id()});
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MaskedFillBackwardNode>(x, mask));
    }
    return out;
}

} // namespace motifcl
