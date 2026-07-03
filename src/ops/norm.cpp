#include <motifcl/ops/norm.hpp>

#include <motifcl/ops/basic_ops.hpp>

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
#include <utility>

namespace motifcl {

namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }
constexpr std::size_t kNormWorkgroup = 256;

bool supports_norm_workgroup(const Backend& backend) {
    return backend.device_info().max_work_group_size >= kNormWorkgroup;
}

bool strict_vulkan_norm_required() {
    const auto enabled = [](const char* value) {
        if (value == nullptr) return false;
        const std::string text(value);
        return text == "1" || text == "true" || text == "TRUE" ||
               text == "on" || text == "ON" || text == "yes" || text == "YES";
    };
    return enabled(std::getenv("MOTIFCL_REQUIRE_VULKAN_COMPUTE")) ||
           enabled(std::getenv("MOTIFCL_REQUIRE_VULKAN_NORM"));
}

void require_rmsnorm_inputs(const Tensor& x, const Tensor& weight, const char* op) {
    MCL_CHECK(x.dtype() == DType::F32 && weight.dtype() == DType::F32, std::string(op) + " supports f32 only");
    MCL_CHECK(x.ndim() == 2 && weight.ndim() == 1, std::string(op) + " expects x [rows, cols] and weight [cols]");
    MCL_CHECK(x.shape()[1] == weight.shape()[0], std::string(op) + " weight shape mismatch");
    MCL_CHECK(x.backend_ptr() == weight.backend_ptr(), std::string(op) + " requires tensors on same backend");
}

bool vulkan_rmsnorm_supported(const Tensor& x, const Tensor& weight, float eps) {
    const bool base = x.dtype() == DType::F32 &&
                      weight.dtype() == DType::F32 &&
                      x.ndim() == 2 &&
                      weight.ndim() == 1 &&
                      x.shape()[0] > 0 &&
                      x.shape()[1] > 0 &&
                      x.shape()[1] == weight.shape()[0] &&
                      x.backend_ptr() == weight.backend_ptr() &&
                      std::isfinite(eps) &&
                      eps > 0.0f;
    if (!base) return false;
    // Vulkan-backed tensors use the cached wave-per-row kernels: any shape,
    // autograd handled by RMSNormBackwardNode.
    if (x.backend().is_vulkan()) return true;
    // Legacy staged host-roundtrip path keeps its original envelope.
    return x.shape()[0] <= 4096 && x.shape()[1] <= 256 && !x.requires_grad() && !weight.requires_grad();
}

Tensor rmsnorm_vulkan_f32(const Tensor& x, const Tensor& weight, const VulkanF32TensorResult& result) {
    MCL_CHECK(result.success, std::string("vulkan rmsnorm f32 failed: ") + result.error);
    MCL_CHECK(result.output.size() == static_cast<std::size_t>(x.numel()),
              "vulkan rmsnorm f32 returned unexpected output size");
    auto out = Tensor::from_cpu(x.backend(), x.shape(), DType::F32, result.output.data());
    autograd::record_op("rmsnorm_vulkan_f32", {x.id(), weight.id()}, {out.id()});
    return out;
}

Tensor rmsnorm_vulkan_f32_device(const Tensor& x, const Tensor& weight, float eps) {
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const auto rows = static_cast<std::size_t>(x.shape()[0]);
    const auto cols = static_cast<std::size_t>(x.shape()[1]);
    const auto result = run_vulkan_rmsnorm(x.backend().vulkan_runtime(),
                                           x.storage().vulkan_buffer,
                                           weight.storage().vulkan_buffer,
                                           out.storage().vulkan_buffer,
                                           rows,
                                           cols,
                                           eps);
    MCL_CHECK(result.success, std::string("vulkan rmsnorm f32 failed: ") + result.error);
    autograd::record_op("rmsnorm_vulkan_f32", {x.id(), weight.id()}, {out.id()});
    return out;
}

struct RMSNormBackwardNode : autograd::Node {
    Tensor x;
    Tensor weight;
    float eps;

    RMSNormBackwardNode(Tensor x_value, Tensor weight_value, float eps_value)
        : x(std::move(x_value)), weight(std::move(weight_value)), eps(eps_value) {}

    std::vector<Tensor> inputs() const override { return {x, weight}; }

    void backward(const Tensor& grad_output) override {
        // The rmsnorm output is often re-viewed by the caller (per-head q/k
        // norm runs on [rows*heads, head_dim] and views back to
        // [rows, heads*head_dim]); the engine then delivers the gradient in
        // the view's shape. Storage is shared and contiguous, so re-view the
        // gradient back to x's shape before the kernel-shape checks.
        const Tensor grad = grad_output.shape() == x.shape()
            ? grad_output
            : grad_output.view(x.shape());
        if (x.requires_grad()) x.backward(rmsnorm_backward_x(x, weight, grad, eps));
        if (weight.requires_grad()) weight.backward(rmsnorm_backward_weight(x, weight, grad, eps));
    }
};
}

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    require_rmsnorm_inputs(x, weight, "rmsnorm");
    const auto selected_backend = selected_norm_backend();
    if ((x.backend().is_vulkan() || selected_backend.kind == MicrokernelBackendKind::Vulkan) &&
        vulkan_rmsnorm_supported(x, weight, eps)) {
        if (x.backend().is_vulkan()) {
            auto out = rmsnorm_vulkan_f32_device(x, weight, eps);
            if (autograd::is_enabled() && (x.requires_grad() || weight.requires_grad())) {
                out.set_requires_grad(true);
                out._set_grad_fn(std::make_shared<RMSNormBackwardNode>(x, weight, eps));
            }
            return out;
        }
        const auto rows = static_cast<std::size_t>(x.shape()[0]);
        const auto cols = static_cast<std::size_t>(x.shape()[1]);
        const auto x_host = x.to_vector<float>();
        const auto weight_host = weight.to_vector<float>();
        const auto result = run_vulkan_rmsnorm(x_host, weight_host, rows, cols, eps);
        if (result.success) return rmsnorm_vulkan_f32(x, weight, result);
        MCL_CHECK(!strict_vulkan_norm_required(),
                  std::string("vulkan rmsnorm f32 failed: ") + result.error);
    }
    MCL_CHECK(!x.backend().is_vulkan(), "vulkan backend does not support this rmsnorm shape");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const bool use_wg = supports_norm_workgroup(x.backend());
    const std::string kernel_name = use_wg ? "rmsnorm_rowwise_wg_f32" : "rmsnorm_rowwise_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, weight.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.set_arg(5, eps);
    if (use_wg) {
        k.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id(), weight.id()}, {out.id()});
    if (autograd::is_enabled() && (x.requires_grad() || weight.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<RMSNormBackwardNode>(x, weight, eps));
    }
    return out;
}

Tensor rmsnorm_backward_x(const Tensor& x, const Tensor& weight, const Tensor& grad_out, float eps) {
    require_rmsnorm_inputs(x, weight, "rmsnorm_backward_x");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == x.shape(), "rmsnorm_backward_x grad_out shape/dtype mismatch");
    if (x.backend().is_vulkan()) {
        auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
        const auto result = run_vulkan_rmsnorm_backward_x(x.backend().vulkan_runtime(),
                                                          x.storage().vulkan_buffer,
                                                          weight.storage().vulkan_buffer,
                                                          grad_out.storage().vulkan_buffer,
                                                          out.storage().vulkan_buffer,
                                                          static_cast<std::size_t>(x.shape()[0]),
                                                          static_cast<std::size_t>(x.shape()[1]),
                                                          eps);
        MCL_CHECK(result.success, std::string("vulkan rmsnorm backward_x failed: ") + result.error);
        autograd::record_op("rmsnorm_backward_x_vulkan_f32", {x.id(), weight.id(), grad_out.id()}, {out.id()});
        return out;
    }
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const bool use_wg = supports_norm_workgroup(x.backend());
    const std::string kernel_name = use_wg ? "rmsnorm_backward_x_wg_f32" : "rmsnorm_backward_x_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, weight.buffer());
    k.set_arg(2, grad_out.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, rows);
    k.set_arg(5, cols);
    k.set_arg(6, eps);
    if (use_wg) {
        k.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);
    } else {
        int n = rows * cols;
        k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id(), weight.id(), grad_out.id()}, {out.id()});
    return out;
}

Tensor rmsnorm_backward_x_residual(const Tensor& x, const Tensor& weight, const Tensor& grad_out,
                                   const Tensor& residual_grad, float eps) {
    require_rmsnorm_inputs(x, weight, "rmsnorm_backward_x_residual");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == x.shape(),
              "rmsnorm_backward_x_residual grad_out shape/dtype mismatch");
    MCL_CHECK(residual_grad.dtype() == DType::F32 && residual_grad.shape() == x.shape(),
              "rmsnorm_backward_x_residual residual_grad shape/dtype mismatch");
    MCL_CHECK(grad_out.backend_ptr() == x.backend_ptr() && residual_grad.backend_ptr() == x.backend_ptr(),
              "rmsnorm_backward_x_residual requires tensors on same backend");
    if (x.backend().is_vulkan()) {
        // Composed as backward_x + residual add on the Vulkan cached path
        // (two dispatches; the fused single-kernel variant stays OpenCL).
        auto norm_grad = rmsnorm_backward_x(x, weight, grad_out, eps);
        return add(norm_grad, residual_grad);
    }
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const bool use_wg = supports_norm_workgroup(x.backend());
    const std::string kernel_name = use_wg ? "rmsnorm_backward_x_residual_wg_f32" : "rmsnorm_backward_x_residual_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, weight.buffer());
    k.set_arg(2, grad_out.buffer());
    k.set_arg(3, residual_grad.buffer());
    k.set_arg(4, out.buffer());
    k.set_arg(5, rows);
    k.set_arg(6, cols);
    k.set_arg(7, eps);
    if (use_wg) {
        k.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);
    } else {
        int n = rows * cols;
        k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id(), weight.id(), grad_out.id(), residual_grad.id()}, {out.id()});
    return out;
}

Tensor rmsnorm_backward_weight(const Tensor& x, const Tensor& weight, const Tensor& grad_out, float eps) {
    require_rmsnorm_inputs(x, weight, "rmsnorm_backward_weight");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == x.shape(), "rmsnorm_backward_weight grad_out shape/dtype mismatch");
    if (x.backend().is_vulkan()) {
        auto out = Tensor::empty(x.backend(), weight.shape(), DType::F32);
        auto row_inv = Tensor::empty(x.backend(), {x.shape()[0]}, DType::F32);
        const auto result = run_vulkan_rmsnorm_backward_weight(x.backend().vulkan_runtime(),
                                                               x.storage().vulkan_buffer,
                                                               grad_out.storage().vulkan_buffer,
                                                               row_inv.storage().vulkan_buffer,
                                                               out.storage().vulkan_buffer,
                                                               static_cast<std::size_t>(x.shape()[0]),
                                                               static_cast<std::size_t>(x.shape()[1]),
                                                               eps);
        MCL_CHECK(result.success, std::string("vulkan rmsnorm backward_weight failed: ") + result.error);
        autograd::record_op("rmsnorm_backward_weight_vulkan_f32", {x.id(), grad_out.id()}, {out.id()});
        return out;
    }
    auto out = Tensor::empty(x.backend(), weight.shape(), DType::F32);
    if (supports_norm_workgroup(x.backend())) {
        int rows = static_cast<int>(x.shape()[0]);
        int cols = static_cast<int>(x.shape()[1]);
        auto row_inv = Tensor::empty(x.backend(), {x.shape()[0]}, DType::F32);

        auto inv_kernel = x.backend().kernels.get("rmsnorm_row_inv_wg_f32");
        inv_kernel.set_arg(0, x.buffer());
        inv_kernel.set_arg(1, row_inv.buffer());
        inv_kernel.set_arg(2, rows);
        inv_kernel.set_arg(3, cols);
        inv_kernel.set_arg(4, eps);
        inv_kernel.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);

        auto grad_kernel = x.backend().kernels.get("rmsnorm_backward_weight_cached_f32");
        grad_kernel.set_arg(0, x.buffer());
        grad_kernel.set_arg(1, grad_out.buffer());
        grad_kernel.set_arg(2, row_inv.buffer());
        grad_kernel.set_arg(3, out.buffer());
        grad_kernel.set_arg(4, rows);
        grad_kernel.set_arg(5, cols);
        grad_kernel.launch1d(round_up(static_cast<std::size_t>(cols), 256), 256);
        autograd::record_replay_op("rmsnorm_backward_weight_cached_f32",
                                   {x.id(), grad_out.id()},
                                   {out.id()},
                                   {row_inv.id()},
                                   []() {});
        return out;
    }
    auto k = x.backend().kernels.get("rmsnorm_backward_weight_f32");
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, grad_out.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.set_arg(5, eps);
    k.launch1d(round_up(static_cast<std::size_t>(cols), 256), 256);
    autograd::record_op("rmsnorm_backward_weight_f32", {x.id(), grad_out.id()}, {out.id()});
    return out;
}

Tensor layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
    MCL_CHECK(x.dtype() == DType::F32 && weight.dtype() == DType::F32 && bias.dtype() == DType::F32, "layernorm supports f32 only");
    MCL_CHECK(x.ndim() == 2 && weight.ndim() == 1 && bias.ndim() == 1, "layernorm expects x [rows, cols], weight [cols], bias [cols]");
    MCL_CHECK(x.shape()[1] == weight.shape()[0] && weight.shape() == bias.shape(), "layernorm parameter shape mismatch");
    MCL_CHECK(x.backend_ptr() == weight.backend_ptr() && x.backend_ptr() == bias.backend_ptr(), "layernorm requires tensors on same backend");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const bool use_wg = supports_norm_workgroup(x.backend());
    const std::string kernel_name = use_wg ? "layernorm_rowwise_wg_f32" : "layernorm_rowwise_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, weight.buffer());
    k.set_arg(2, bias.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, rows);
    k.set_arg(5, cols);
    k.set_arg(6, eps);
    if (use_wg) {
        k.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id(), weight.id(), bias.id()}, {out.id()});
    return out;
}

} // namespace motifcl
