#include <motifcl/ops/loss.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/indexing.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

namespace motifcl {

namespace {
constexpr int kChunk = 256;
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }

bool vulkan_ce_cache_enabled() {
    const char* env = std::getenv("MOTIFCL_VK_CE_CACHE");
    return !(env && (std::string(env) == "0" || std::string(env) == "false" || std::string(env) == "FALSE"));
}

Tensor mean_reduce_gpu(const Tensor& partial, int denominator) {
    auto out = Tensor::empty(partial.backend(), {}, DType::F32);
    auto k = partial.backend().kernels.get("mean_reduce_f32");
    const int partial_count = static_cast<int>(partial.numel());
    k.set_arg(0, partial.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, partial_count);
    k.set_arg(3, denominator);
    k.launch1d(256, 256);
    return out;
}

struct MSEBackwardNode : autograd::Node {
    Tensor pred, target;
    MSEBackwardNode(Tensor pred, Tensor target) : pred(std::move(pred)), target(std::move(target)) {}
    std::vector<Tensor> inputs() const override { return {pred, target}; }
    void backward(const Tensor& grad_output) override {
        if (pred.requires_grad()) pred.backward(mse_backward_op(pred, target, grad_output));
    }
};

struct SoftmaxCrossEntropyBackwardNode : autograd::Node {
    Tensor logits, targets, row_stats;
    SoftmaxCrossEntropyBackwardNode(Tensor logits, Tensor targets, Tensor row_stats = {})
        : logits(std::move(logits)), targets(std::move(targets)), row_stats(std::move(row_stats)) {}
    std::vector<Tensor> inputs() const override { return {logits}; }
    void backward(const Tensor& grad_output) override {
        if (!logits.requires_grad()) return;
        if (!row_stats.valid()) {
            logits.backward(softmax_cross_entropy_backward_op(logits, targets, grad_output));
            return;
        }
        MCL_CHECK(grad_output.dtype() == DType::F32 && grad_output.numel() == 1,
                  "softmax_cross_entropy_backward grad_out must be an f32 scalar");
        MCL_CHECK(logits.backend_ptr() == grad_output.backend_ptr(),
                  "softmax_cross_entropy_backward requires one backend");
        auto grad_logits = Tensor::empty(logits.backend(), logits.shape(), DType::F32);
        const auto run = run_vulkan_softmax_cross_entropy_backward_cached(
            logits.backend().vulkan_runtime(), logits.storage().vulkan_buffer,
            targets.storage().vulkan_buffer, grad_output.storage().vulkan_buffer,
            row_stats.storage().vulkan_buffer, grad_logits.storage().vulkan_buffer,
            static_cast<std::size_t>(logits.shape()[0]), static_cast<std::size_t>(logits.shape()[1]));
        MCL_CHECK(run.success, std::string("vulkan cached softmax_cross_entropy backward failed: ") + run.error);
        autograd::record_op("softmax_cross_entropy_backward_cached_vulkan_f32",
                            {logits.id(), targets.id(), grad_output.id(), row_stats.id()}, {grad_logits.id()});
        logits.backward(grad_logits);
    }
};

} // namespace

// Fused head+CE node: the logits tensor is produced outside the graph (the node owns
// both head gradients), and the backward pass recomputes dlogits inside the two head
// GEMMs from logits + row stats + targets, so the [rows, classes] buffer is never built.
struct HeadCEFusedNode : autograd::Node {
    Tensor xf, weight, logits, targets, row_stats;
    HeadCEFusedNode(Tensor xf, Tensor weight, Tensor logits, Tensor targets, Tensor row_stats)
        : xf(std::move(xf)), weight(std::move(weight)), logits(std::move(logits)),
          targets(std::move(targets)), row_stats(std::move(row_stats)) {}
    std::vector<Tensor> inputs() const override { return {xf, weight}; }
    void backward(const Tensor& grad_output) override {
        MCL_CHECK(grad_output.dtype() == DType::F32 && grad_output.numel() == 1,
                  "head_ce_fused grad_out must be an f32 scalar");
        MCL_CHECK(logits.ndim() == 2 && xf.ndim() == 2 && weight.ndim() == 2,
                  "head_ce_fused backward expects rank-2 tensors");
        auto& backend = xf.backend();
        auto& runtime = backend.vulkan_runtime();
        const auto rows = static_cast<std::size_t>(logits.shape()[0]);
        const auto classes = static_cast<std::size_t>(logits.shape()[1]);
        const auto in_features = static_cast<std::size_t>(xf.shape()[1]);
        if (xf.requires_grad()) {
            auto grad_xf = Tensor::empty(backend, xf.shape(), DType::F32);
            const auto run = run_vulkan_matmul_ce_nt(
                runtime, logits.storage().vulkan_buffer, weight.storage().vulkan_buffer,
                row_stats.storage().vulkan_buffer, targets.storage().vulkan_buffer,
                grad_output.storage().vulkan_buffer, grad_xf.storage().vulkan_buffer, rows, classes,
                in_features);
            MCL_CHECK(run.success, std::string("vulkan fused head grad_input failed: ") + run.error);
            autograd::record_op("head_ce_fused_grad_input_vulkan_f32",
                                {logits.id(), weight.id(), targets.id(), grad_output.id(), row_stats.id()},
                                {grad_xf.id()});
            xf.backward(grad_xf);
        }
        if (weight.requires_grad()) {
            auto grad_w = Tensor::empty(backend, weight.shape(), DType::F32);
            const auto run = run_vulkan_matmul_ce_tn(
                runtime, xf.storage().vulkan_buffer, logits.storage().vulkan_buffer,
                row_stats.storage().vulkan_buffer, targets.storage().vulkan_buffer,
                grad_output.storage().vulkan_buffer, grad_w.storage().vulkan_buffer, in_features, rows,
                classes);
            MCL_CHECK(run.success, std::string("vulkan fused head grad_weight failed: ") + run.error);
            autograd::record_op("head_ce_fused_grad_weight_vulkan_f32",
                                {xf.id(), logits.id(), targets.id(), grad_output.id(), row_stats.id()},
                                {grad_w.id()});
            weight.backward(grad_w);
        }
    }
};

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    MCL_CHECK(pred.dtype() == DType::F32 && target.dtype() == DType::F32, "mse_loss supports f32 only");
    MCL_CHECK(pred.shape() == target.shape(), "mse_loss shape mismatch");
    if (pred.backend().is_vulkan()) {
        // Vulkan-native: mean((pred-target)^2) via sub/mul/mean_all. Gradients
        // flow through the composed autograd chain (mean -> mul -> sub).
        auto diff = sub(pred, target);
        return mean_all(mul(diff, diff));
    }
    int n = static_cast<int>(pred.numel());
    int chunks = (n + kChunk - 1) / kChunk;
    auto partial = Tensor::empty(pred.backend(), {chunks}, DType::F32);
    auto k = pred.backend().kernels.get("mse_loss_partial_f32");
    k.set_arg(0, pred.buffer());
    k.set_arg(1, target.buffer());
    k.set_arg(2, partial.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(chunks), 256), 256);

    auto out = mean_reduce_gpu(partial, n);
    autograd::record_replay_op("mse_loss", {pred.id(), target.id()}, {out.id()}, {partial.id()}, []() {});
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
    auto out = Tensor::empty(pred.backend(), pred.shape(), DType::F32);
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
    if (logits.backend().is_vulkan()) {
        auto partial = Tensor::empty(logits.backend(), {rows}, DType::F32);
        auto out = Tensor::empty(logits.backend(), {}, DType::F32);
        const bool needs_backward = autograd::is_enabled() && logits.requires_grad();
        auto& runtime = logits.backend().vulkan_runtime();
        const bool use_cache = needs_backward && vulkan_ce_cache_enabled() &&
                               runtime.caps().max_workgroup_invocations >= 128;
        Tensor row_stats;
        VulkanOpResult run;
        if (use_cache) {
            row_stats = Tensor::empty(logits.backend(), {rows, 2}, DType::F32);
            run = run_vulkan_softmax_cross_entropy_cached(
                runtime, logits.storage().vulkan_buffer, targets.storage().vulkan_buffer,
                partial.storage().vulkan_buffer, row_stats.storage().vulkan_buffer,
                out.storage().vulkan_buffer, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        } else {
            run = run_vulkan_softmax_cross_entropy(
                runtime, logits.storage().vulkan_buffer, targets.storage().vulkan_buffer,
                partial.storage().vulkan_buffer, out.storage().vulkan_buffer,
                static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        }
        MCL_CHECK(run.success, std::string("vulkan softmax_cross_entropy failed: ") + run.error);
        if (use_cache) {
            autograd::record_op("softmax_cross_entropy_cached_vulkan_f32",
                                {logits.id(), targets.id()}, {out.id(), row_stats.id()});
        } else {
            autograd::record_op("softmax_cross_entropy_vulkan_f32", {logits.id(), targets.id()}, {out.id()});
        }
        if (needs_backward) {
            out.set_requires_grad(true);
            out._set_grad_fn(std::make_shared<SoftmaxCrossEntropyBackwardNode>(logits, targets, std::move(row_stats)));
        }
        return out;
    }
    auto partial = Tensor::empty(logits.backend(), {rows}, DType::F32);
    auto k = logits.backend().kernels.get("softmax_cross_entropy_partial_f32_i32");
    k.set_arg(0, logits.buffer());
    k.set_arg(1, targets.buffer());
    k.set_arg(2, partial.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);

    auto out = mean_reduce_gpu(partial, rows);
    autograd::record_replay_op("softmax_cross_entropy", {logits.id(), targets.id()}, {out.id()}, {partial.id()}, []() {});
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
    auto out = Tensor::empty(logits.backend(), logits.shape(), DType::F32);
    const int rows = static_cast<int>(logits.shape()[0]);
    const int cols = static_cast<int>(logits.shape()[1]);
    if (logits.backend().is_vulkan()) {
        const auto run = run_vulkan_softmax_cross_entropy_backward(logits.backend().vulkan_runtime(),
                                                                   logits.storage().vulkan_buffer,
                                                                   targets.storage().vulkan_buffer,
                                                                   grad_out.storage().vulkan_buffer,
                                                                   out.storage().vulkan_buffer,
                                                                   static_cast<std::size_t>(rows),
                                                                   static_cast<std::size_t>(cols));
        MCL_CHECK(run.success, std::string("vulkan softmax_cross_entropy backward failed: ") + run.error);
        autograd::record_op("softmax_cross_entropy_backward_vulkan_f32",
                            {logits.id(), targets.id(), grad_out.id()}, {out.id()});
        return out;
    }
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

Tensor head_ce_fused(const Tensor& xf, const Tensor& weight, const Tensor& targets) {
    MCL_CHECK(xf.dtype() == DType::F32 && weight.dtype() == DType::F32 && targets.dtype() == DType::I32,
              "head_ce_fused expects f32 activations/weights and i32 targets");
    MCL_CHECK(xf.ndim() == 2 && weight.ndim() == 2, "head_ce_fused expects rank-2 tensors");
    MCL_CHECK(targets.numel() == xf.shape()[0], "head_ce_fused needs one target per row");
    MCL_CHECK(xf.backend_ptr() == weight.backend_ptr() && xf.backend_ptr() == targets.backend_ptr(),
              "head_ce_fused requires one backend");
    auto& backend = xf.backend();
    MCL_CHECK(backend.is_vulkan(), "head_ce_fused requires the Vulkan backend");
    const int rows = static_cast<int>(xf.shape()[0]);
    const int in_features = static_cast<int>(xf.shape()[1]);
    // The head weight is stored [in, classes] (it feeds matmul(xf, weight), the nn layout).
    MCL_CHECK(weight.shape()[0] == in_features, "head_ce_fused weight width mismatch");
    const int classes = static_cast<int>(weight.shape()[1]);
    MCL_CHECK(rows % 64 == 0 && classes % 64 == 0 && in_features % 4 == 0,
              "head_ce_fused needs rows/classes multiples of 64 and in_features a multiple of four");

    Tensor logits, row_stats, partial, out;
    {
        // The logits are inputs to the fused node, not a graph node: the node owns both
        // head gradients, so the matmul must not also record a backward.
        autograd::NoGradGuard guard;
        logits = matmul(xf, weight);
        row_stats = Tensor::empty(backend, {rows, 2}, DType::F32);
        partial = Tensor::empty(backend, {rows}, DType::F32);
        out = Tensor::empty(backend, {}, DType::F32);
        const auto run = run_vulkan_softmax_cross_entropy_cached(
            backend.vulkan_runtime(), logits.storage().vulkan_buffer, targets.storage().vulkan_buffer,
            partial.storage().vulkan_buffer, row_stats.storage().vulkan_buffer, out.storage().vulkan_buffer,
            static_cast<std::size_t>(rows), static_cast<std::size_t>(classes));
        MCL_CHECK(run.success, std::string("vulkan fused head cross-entropy failed: ") + run.error);
    }
    autograd::record_op("head_ce_fused_vulkan_f32", {xf.id(), weight.id(), targets.id()},
                        {out.id(), logits.id(), row_stats.id()});
    if (autograd::is_enabled() && (xf.requires_grad() || weight.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(
            std::make_shared<HeadCEFusedNode>(xf, weight, logits, targets, std::move(row_stats)));
    }
    return out;
}

} // namespace motifcl
