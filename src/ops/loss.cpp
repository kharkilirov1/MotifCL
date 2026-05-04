#include <motifcl/ops/loss.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cmath>
#include <memory>
#include <vector>

namespace motifcl {

namespace {
constexpr int kChunk = 256;
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }

float mean_partial_f32(const Tensor& partial, int denominator) {
    auto host = partial.to_vector<float>();
    double sum = 0.0;
    for (float v : host) sum += v;
    return denominator > 0 ? static_cast<float>(sum / static_cast<double>(denominator)) : 0.0f;
}

void upload_scalar_f32(const Tensor& out, float value) {
    out.buffer().upload(&value, sizeof(float), out.offset());
}

struct MSEBackwardNode : autograd::Node {
    Tensor pred, target;
    MSEBackwardNode(Tensor pred, Tensor target) : pred(std::move(pred)), target(std::move(target)) {}
    void backward(const Tensor& grad_output) override {
        if (pred.requires_grad()) pred.backward(mse_backward_op(pred, target, grad_output));
    }
};

struct SoftmaxCrossEntropyBackwardNode : autograd::Node {
    Tensor logits, targets;
    SoftmaxCrossEntropyBackwardNode(Tensor logits, Tensor targets) : logits(std::move(logits)), targets(std::move(targets)) {}
    void backward(const Tensor& grad_output) override {
        if (logits.requires_grad()) logits.backward(softmax_cross_entropy_backward_op(logits, targets, grad_output));
    }
};

} // namespace

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    MCL_CHECK(pred.dtype() == DType::F32 && target.dtype() == DType::F32, "mse_loss supports f32 only");
    MCL_CHECK(pred.shape() == target.shape(), "mse_loss shape mismatch");
    int n = static_cast<int>(pred.numel());
    int chunks = (n + kChunk - 1) / kChunk;
    auto partial = Tensor::zeros(pred.backend(), {chunks}, DType::F32);
    auto k = pred.backend().kernels.get("mse_loss_partial_f32");
    k.set_arg(0, pred.buffer());
    k.set_arg(1, target.buffer());
    k.set_arg(2, partial.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(chunks), 256), 256);

    float mean = mean_partial_f32(partial, n);
    auto out = Tensor::from_cpu(pred.backend(), {}, DType::F32, &mean);
    autograd::record_replay_op("mse_loss", {pred.id(), target.id()}, {out.id()}, {partial.id()}, [partial, out, n]() {
        upload_scalar_f32(out, mean_partial_f32(partial, n));
    });
    if (autograd::is_enabled() && pred.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MSEBackwardNode>(pred, target));
    }
    return out;
}

Tensor mse_backward_op(const Tensor& pred, const Tensor& target, const Tensor& grad_out) {
    MCL_CHECK(pred.dtype() == DType::F32 && target.dtype() == DType::F32 && grad_out.dtype() == DType::F32, "mse_backward supports f32 only");
    MCL_CHECK(pred.shape() == target.shape(), "mse_backward shape mismatch");
    MCL_CHECK(grad_out.numel() == 1, "mse_backward grad_out must be scalar");
    auto out = Tensor::zeros(pred.backend(), pred.shape(), DType::F32);
    auto k = pred.backend().kernels.get("mse_backward_f32");
    int n = static_cast<int>(pred.numel());
    k.set_arg(0, pred.buffer());
    k.set_arg(1, target.buffer());
    k.set_arg(2, grad_out.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("mse_backward_f32", {pred.id(), target.id(), grad_out.id()}, {out.id()});
    return out;
}

Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets) {
    MCL_CHECK(logits.dtype() == DType::F32, "softmax_cross_entropy logits must be f32");
    MCL_CHECK(targets.dtype() == DType::I32, "softmax_cross_entropy targets must be i32 class ids");
    MCL_CHECK(logits.ndim() == 2, "softmax_cross_entropy expects logits [rows, classes]");
    MCL_CHECK(targets.numel() == logits.shape()[0], "softmax_cross_entropy targets must have one class id per row");
    MCL_CHECK(logits.backend_ptr() == targets.backend_ptr(), "softmax_cross_entropy requires one backend");

    const int rows = static_cast<int>(logits.shape()[0]);
    const int cols = static_cast<int>(logits.shape()[1]);
    auto partial = Tensor::zeros(logits.backend(), {rows}, DType::F32);
    auto k = logits.backend().kernels.get("softmax_cross_entropy_partial_f32_i32");
    k.set_arg(0, logits.buffer());
    k.set_arg(1, targets.buffer());
    k.set_arg(2, partial.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);

    float mean = mean_partial_f32(partial, rows);
    auto out = Tensor::from_cpu(logits.backend(), {}, DType::F32, &mean);
    autograd::record_replay_op("softmax_cross_entropy", {logits.id(), targets.id()}, {out.id()}, {partial.id()}, [partial, out, rows]() {
        upload_scalar_f32(out, mean_partial_f32(partial, rows));
    });
    if (autograd::is_enabled() && logits.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<SoftmaxCrossEntropyBackwardNode>(logits, targets));
    }
    return out;
}

Tensor softmax_cross_entropy_backward_op(const Tensor& logits, const Tensor& targets, const Tensor& grad_out) {
    MCL_CHECK(logits.dtype() == DType::F32 && targets.dtype() == DType::I32 && grad_out.dtype() == DType::F32, "softmax_cross_entropy_backward dtype mismatch");
    MCL_CHECK(logits.ndim() == 2, "softmax_cross_entropy_backward expects logits [rows, classes]");
    MCL_CHECK(targets.numel() == logits.shape()[0], "softmax_cross_entropy_backward targets shape mismatch");
    MCL_CHECK(grad_out.numel() == 1, "softmax_cross_entropy_backward grad_out must be scalar");
    auto out = Tensor::zeros(logits.backend(), logits.shape(), DType::F32);
    const int rows = static_cast<int>(logits.shape()[0]);
    const int cols = static_cast<int>(logits.shape()[1]);
    auto k = logits.backend().kernels.get("softmax_cross_entropy_backward_f32_i32");
    k.set_arg(0, logits.buffer());
    k.set_arg(1, targets.buffer());
    k.set_arg(2, grad_out.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, rows);
    k.set_arg(5, cols);
    k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    autograd::record_op("softmax_cross_entropy_backward_f32_i32", {logits.id(), targets.id(), grad_out.id()}, {out.id()});
    return out;
}

} // namespace motifcl
