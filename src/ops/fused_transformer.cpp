#include <motifcl/ops/fused_transformer.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/norm.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace motifcl {

namespace {

constexpr std::size_t kNormWorkgroup = 256;

std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

bool supports_norm_workgroup(const Backend& backend) {
    return backend.device_info().max_work_group_size >= kNormWorkgroup;
}

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return std::string(value) != "0" && std::string(value) != "false" && std::string(value) != "FALSE";
}

void launch_register_block4(Kernel& k, int M, int N) {
    constexpr std::size_t kBlock = 32;
    constexpr std::size_t kLocal2D = 8;
    k.launch2d((round_up(static_cast<std::size_t>(N), kBlock) / 4),
               (round_up(static_cast<std::size_t>(M), kBlock) / 4),
               kLocal2D,
               kLocal2D);
}

void require_fused_mlp_inputs(const Tensor& x,
                              const Tensor& norm_weight,
                              const Tensor& gate_up_weight,
                              const Tensor& down_weight,
                              const char* op) {
    MCL_CHECK(x.dtype() == DType::F32 && norm_weight.dtype() == DType::F32 &&
                  gate_up_weight.dtype() == DType::F32 && down_weight.dtype() == DType::F32,
              std::string(op) + " supports f32 only");
    MCL_CHECK(x.ndim() == 2 && norm_weight.ndim() == 1 &&
                  gate_up_weight.ndim() == 2 && down_weight.ndim() == 2,
              std::string(op) + " expects x [rows, channels], norm_weight [channels], "
                                "gate_up_weight [channels, 2*hidden], down_weight [hidden, channels]");
    MCL_CHECK(x.shape()[1] == norm_weight.shape()[0],
              std::string(op) + " norm weight shape mismatch");
    MCL_CHECK(gate_up_weight.shape()[0] == x.shape()[1] && gate_up_weight.shape()[1] % 2 == 0,
              std::string(op) + " gate/up weight shape mismatch");
    const int64_t hidden = gate_up_weight.shape()[1] / 2;
    MCL_CHECK(down_weight.shape()[0] == hidden && down_weight.shape()[1] == x.shape()[1],
              std::string(op) + " down weight shape mismatch");
    MCL_CHECK(x.backend_ptr() == norm_weight.backend_ptr() &&
                  x.backend_ptr() == gate_up_weight.backend_ptr() &&
                  x.backend_ptr() == down_weight.backend_ptr(),
              std::string(op) + " requires all tensors on same backend");
}

void require_rmsnorm_pair_inputs(const Tensor& x, const Tensor& weight, const char* op) {
    MCL_CHECK(x.dtype() == DType::F32 && weight.dtype() == DType::F32,
              std::string(op) + " supports f32 only");
    MCL_CHECK(x.ndim() == 2 && weight.ndim() == 1,
              std::string(op) + " expects x [rows, cols] and weight [cols]");
    MCL_CHECK(x.shape()[1] == weight.shape()[0],
              std::string(op) + " weight shape mismatch");
    MCL_CHECK(x.backend_ptr() == weight.backend_ptr(),
              std::string(op) + " requires tensors on same backend");
}

void require_swiglu_down_backward_inputs(const Tensor& packed,
                                         const Tensor& down_weight,
                                         const Tensor& grad_out,
                                         const char* op) {
    MCL_CHECK(packed.dtype() == DType::F32 && down_weight.dtype() == DType::F32 &&
                  grad_out.dtype() == DType::F32,
              std::string(op) + " supports f32 only");
    MCL_CHECK(packed.ndim() == 2 && down_weight.ndim() == 2 && grad_out.ndim() == 2,
              std::string(op) + " expects packed [rows, 2*hidden], down_weight [hidden, channels], "
                                "grad_out [rows, channels]");
    MCL_CHECK(packed.shape()[1] % 2 == 0, std::string(op) + " packed second dim must be even");
    const int64_t rows = packed.shape()[0];
    const int64_t hidden = packed.shape()[1] / 2;
    const int64_t channels = down_weight.shape()[1];
    MCL_CHECK(down_weight.shape()[0] == hidden, std::string(op) + " hidden dimension mismatch");
    MCL_CHECK(grad_out.shape() == Shape({rows, channels}), std::string(op) + " grad_out shape mismatch");
    MCL_CHECK(packed.backend_ptr() == down_weight.backend_ptr() &&
                  packed.backend_ptr() == grad_out.backend_ptr(),
              std::string(op) + " requires all tensors on same backend");
}

void require_row_inv_for_x(const Tensor& x, const Tensor& row_inv, const char* op) {
    MCL_CHECK(row_inv.dtype() == DType::F32 && row_inv.ndim() == 1,
              std::string(op) + " row_inv must be f32 [rows]");
    MCL_CHECK(x.ndim() == 2 && row_inv.shape()[0] == x.shape()[0],
              std::string(op) + " row_inv shape mismatch");
    MCL_CHECK(row_inv.backend_ptr() == x.backend_ptr(),
              std::string(op) + " requires row_inv on same backend");
}

Tensor rmsnorm_row_inv_cached(const Tensor& x, float eps) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "rmsnorm_row_inv_cached expects f32 [rows, cols]");
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);
    auto row_inv = Tensor::empty(x.backend(), {x.shape()[0]}, DType::F32);
    const bool use_wg = supports_norm_workgroup(x.backend());
    const std::string kernel_name = use_wg ? "rmsnorm_row_inv_wg_f32" : "rmsnorm_row_inv_f32";
    auto k = x.backend().kernels.get(kernel_name);
    k.set_arg(0, x.buffer());
    k.set_arg(1, row_inv.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    k.set_arg(4, eps);
    if (use_wg) {
        k.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id()}, {row_inv.id()});
    return row_inv;
}

Tensor rmsnorm_matmul_forward_cached(const Tensor& x,
                                     const Tensor& norm_weight,
                                     const Tensor& row_inv,
                                     const Tensor& weight) {
    require_rmsnorm_pair_inputs(x, norm_weight, "rmsnorm_matmul_forward_cached");
    require_row_inv_for_x(x, row_inv, "rmsnorm_matmul_forward_cached");
    MCL_CHECK(weight.dtype() == DType::F32 && weight.ndim() == 2 &&
                  weight.shape()[0] == x.shape()[1] &&
                  weight.backend_ptr() == x.backend_ptr(),
              "rmsnorm_matmul_forward_cached weight shape/dtype/backend mismatch");
    auto out = Tensor::empty(x.backend(), {x.shape()[0], weight.shape()[1]}, DType::F32);
    auto k = x.backend().kernels.get("rmsnorm_matmul_rb4_f32");
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);
    const int out_cols = static_cast<int>(weight.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, norm_weight.buffer());
    k.set_arg(2, row_inv.buffer());
    k.set_arg(3, weight.buffer());
    k.set_arg(4, out.buffer());
    k.set_arg(5, rows);
    k.set_arg(6, out_cols);
    k.set_arg(7, cols);
    launch_register_block4(k, rows, out_cols);
    autograd::record_op("rmsnorm_matmul_rb4_f32", {x.id(), norm_weight.id(), row_inv.id(), weight.id()}, {out.id()});
    return out;
}

Tensor rmsnorm_matmul_transpose_a_cached(const Tensor& x,
                                         const Tensor& norm_weight,
                                         const Tensor& row_inv,
                                         const Tensor& grad_packed) {
    require_rmsnorm_pair_inputs(x, norm_weight, "rmsnorm_matmul_transpose_a_cached");
    require_row_inv_for_x(x, row_inv, "rmsnorm_matmul_transpose_a_cached");
    MCL_CHECK(grad_packed.dtype() == DType::F32 && grad_packed.ndim() == 2 &&
                  grad_packed.shape()[0] == x.shape()[0],
              "rmsnorm_matmul_transpose_a_cached grad_packed shape/dtype mismatch");
    MCL_CHECK(grad_packed.backend_ptr() == x.backend_ptr(),
              "rmsnorm_matmul_transpose_a_cached requires tensors on same backend");
    auto out = Tensor::empty(x.backend(), {x.shape()[1], grad_packed.shape()[1]}, DType::F32);
    auto k = x.backend().kernels.get("rmsnorm_matmul_transa_rb4_f32");
    const int out_rows = static_cast<int>(x.shape()[1]);
    const int out_cols = static_cast<int>(grad_packed.shape()[1]);
    const int reduce_rows = static_cast<int>(x.shape()[0]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, norm_weight.buffer());
    k.set_arg(2, row_inv.buffer());
    k.set_arg(3, grad_packed.buffer());
    k.set_arg(4, out.buffer());
    k.set_arg(5, out_rows);
    k.set_arg(6, out_cols);
    k.set_arg(7, reduce_rows);
    launch_register_block4(k, out_rows, out_cols);
    autograd::record_op("rmsnorm_matmul_transa_rb4_f32",
                        {x.id(), norm_weight.id(), row_inv.id(), grad_packed.id()},
                        {out.id()});
    return out;
}

Tensor rmsnorm_backward_x_residual_cached(const Tensor& x,
                                          const Tensor& weight,
                                          const Tensor& row_inv,
                                          const Tensor& grad_out,
                                          const Tensor& residual_grad) {
    require_rmsnorm_pair_inputs(x, weight, "rmsnorm_backward_x_residual_cached");
    require_row_inv_for_x(x, row_inv, "rmsnorm_backward_x_residual_cached");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == x.shape(),
              "rmsnorm_backward_x_residual_cached grad_out shape/dtype mismatch");
    MCL_CHECK(residual_grad.dtype() == DType::F32 && residual_grad.shape() == x.shape(),
              "rmsnorm_backward_x_residual_cached residual_grad shape/dtype mismatch");
    MCL_CHECK(grad_out.backend_ptr() == x.backend_ptr() && residual_grad.backend_ptr() == x.backend_ptr(),
              "rmsnorm_backward_x_residual_cached requires tensors on same backend");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    const bool use_wg = supports_norm_workgroup(x.backend());
    const std::string kernel_name = use_wg ? "rmsnorm_backward_x_residual_cached_wg_f32"
                                           : "rmsnorm_backward_x_residual_cached_f32";
    auto k = x.backend().kernels.get(kernel_name);
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, weight.buffer());
    k.set_arg(2, row_inv.buffer());
    k.set_arg(3, grad_out.buffer());
    k.set_arg(4, residual_grad.buffer());
    k.set_arg(5, out.buffer());
    k.set_arg(6, rows);
    k.set_arg(7, cols);
    if (use_wg) {
        k.launch2d(kNormWorkgroup, static_cast<std::size_t>(rows), kNormWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows * cols), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id(), weight.id(), row_inv.id(), grad_out.id(), residual_grad.id()}, {out.id()});
    return out;
}

Tensor rmsnorm_backward_weight_cached_from_inv(const Tensor& x,
                                               const Tensor& row_inv,
                                               const Tensor& grad_out) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "rmsnorm_backward_weight_cached_from_inv expects x f32 [rows, cols]");
    require_row_inv_for_x(x, row_inv, "rmsnorm_backward_weight_cached_from_inv");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == x.shape(),
              "rmsnorm_backward_weight_cached_from_inv grad_out shape/dtype mismatch");
    MCL_CHECK(grad_out.backend_ptr() == x.backend_ptr(),
              "rmsnorm_backward_weight_cached_from_inv requires tensors on same backend");
    auto out = Tensor::empty(x.backend(), {x.shape()[1]}, DType::F32);
    auto k = x.backend().kernels.get("rmsnorm_backward_weight_cached_f32");
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, grad_out.buffer());
    k.set_arg(2, row_inv.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, rows);
    k.set_arg(5, cols);
    k.launch1d(round_up(static_cast<std::size_t>(cols), 256), 256);
    autograd::record_op("rmsnorm_backward_weight_cached_f32", {x.id(), grad_out.id(), row_inv.id()}, {out.id()});
    return out;
}

struct FusedSwiGLUMLPRMSNormResidualBackwardNode : autograd::Node {
    Tensor x;
    Tensor norm_weight;
    Tensor gate_up_weight;
    Tensor down_weight;
    Tensor normed;
    Tensor row_inv;
    Tensor packed;
    Tensor hidden;
    float eps = 1e-6f;
    bool high_level = false;

    FusedSwiGLUMLPRMSNormResidualBackwardNode(Tensor x_value,
                                              Tensor norm_weight_value,
                                              Tensor gate_up_weight_value,
                                              Tensor down_weight_value,
                                              Tensor normed_value,
                                              Tensor row_inv_value,
                                              Tensor packed_value,
                                              Tensor hidden_value,
                                              float eps_value,
                                              bool high_level_value)
        : x(std::move(x_value)),
          norm_weight(std::move(norm_weight_value)),
          gate_up_weight(std::move(gate_up_weight_value)),
          down_weight(std::move(down_weight_value)),
          normed(std::move(normed_value)),
          row_inv(std::move(row_inv_value)),
          packed(std::move(packed_value)),
          hidden(std::move(hidden_value)),
          eps(eps_value),
          high_level(high_level_value) {}

    std::vector<Tensor> inputs() const override {
        return {x, norm_weight, gate_up_weight, down_weight};
    }

    void backward(const Tensor& grad_output) override {
        MCL_CHECK(grad_output.dtype() == DType::F32 && grad_output.shape() == x.shape(),
                  "fused_swiglu_mlp_rmsnorm_residual backward grad_output shape/dtype mismatch");

        if (down_weight.requires_grad()) {
            down_weight.backward(matmul_transpose_a(hidden, grad_output));
        }

        const Tensor grad_packed = swiglu_down_backward_packed(packed, down_weight, grad_output);

        if (gate_up_weight.requires_grad()) {
            gate_up_weight.backward(high_level
                                        ? rmsnorm_matmul_transpose_a_cached(x, norm_weight, row_inv, grad_packed)
                                        : matmul_transpose_a(normed, grad_packed));
        }

        const Tensor grad_normed = matmul_transpose_b(grad_packed, gate_up_weight);
        if (x.requires_grad()) {
            x.backward(high_level
                           ? rmsnorm_backward_x_residual_cached(x, norm_weight, row_inv, grad_normed, grad_output)
                           : rmsnorm_backward_x_residual(x, norm_weight, grad_normed, grad_output, eps));
        }
        if (norm_weight.requires_grad()) {
            norm_weight.backward(high_level
                                     ? rmsnorm_backward_weight_cached_from_inv(x, row_inv, grad_normed)
                                     : rmsnorm_backward_weight(x, norm_weight, grad_normed, eps));
        }
    }
};

} // namespace

Tensor swiglu_down_backward_packed(const Tensor& packed, const Tensor& down_weight, const Tensor& grad_out) {
    require_swiglu_down_backward_inputs(packed, down_weight, grad_out, "swiglu_down_backward_packed");
    const int rows = static_cast<int>(packed.shape()[0]);
    const int hidden = static_cast<int>(packed.shape()[1] / 2);
    const int channels = static_cast<int>(down_weight.shape()[1]);

    auto grad_packed = Tensor::empty(packed.backend(), packed.shape(), DType::F32);
    auto k = packed.backend().kernels.get("swiglu_down_backward_packed_f32");
    k.set_arg(0, packed.buffer());
    k.set_arg(1, down_weight.buffer());
    k.set_arg(2, grad_out.buffer());
    k.set_arg(3, grad_packed.buffer());
    k.set_arg(4, rows);
    k.set_arg(5, hidden);
    k.set_arg(6, channels);
    launch_register_block4(k, rows, hidden);
    autograd::record_op("swiglu_down_backward_packed_f32",
                        {packed.id(), down_weight.id(), grad_out.id()},
                        {grad_packed.id()});
    return grad_packed;
}

Tensor fused_swiglu_mlp_rmsnorm_residual(const Tensor& x,
                                         const Tensor& norm_weight,
                                         const Tensor& gate_up_weight,
                                         const Tensor& down_weight,
                                         float eps) {
    require_fused_mlp_inputs(x, norm_weight, gate_up_weight, down_weight,
                             "fused_swiglu_mlp_rmsnorm_residual");

    const bool high_level = env_enabled("MOTIFCL_ENABLE_HIGH_LEVEL_MLP_FUSION");
    Tensor normed;
    Tensor row_inv;
    Tensor packed;
    Tensor hidden;
    Tensor mlp_out;
    Tensor out;
    {
        autograd::NoGradGuard guard;
        if (high_level) {
            row_inv = rmsnorm_row_inv_cached(x, eps);
            packed = rmsnorm_matmul_forward_cached(x, norm_weight, row_inv, gate_up_weight);
        } else {
            normed = rmsnorm(x, norm_weight, eps);
            packed = matmul(normed, gate_up_weight);
        }
        hidden = swiglu(packed);
        mlp_out = matmul(hidden, down_weight);
        out = add(x, mlp_out);
    }

    if (autograd::is_enabled() &&
        (x.requires_grad() || norm_weight.requires_grad() ||
         gate_up_weight.requires_grad() || down_weight.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<FusedSwiGLUMLPRMSNormResidualBackwardNode>(
            x, norm_weight, gate_up_weight, down_weight, normed, row_inv, packed, hidden, eps, high_level));
    }
    return out;
}

} // namespace motifcl
