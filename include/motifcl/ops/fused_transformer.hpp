#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

// Fuses the down-projection input gradient with SwiGLU backward:
//   grad_hidden = grad_out @ down_weight^T
//   grad_packed = d_swiglu(packed, grad_hidden)
// packed: [rows, 2 * hidden], down_weight: [hidden, channels], grad_out: [rows, channels].
Tensor swiglu_down_backward_packed(const Tensor& packed, const Tensor& down_weight, const Tensor& grad_out);

// Transformer MLP branch fusion for the common modern block:
//   out = x + (swiglu(rmsnorm(x, norm_weight, eps) @ gate_up_weight) @ down_weight)
// Backward is attached as one custom autograd node and uses fused kernels for
// SwiGLU+down-input backward and RMSNorm+residual backward.
Tensor fused_swiglu_mlp_rmsnorm_residual(const Tensor& x,
                                         const Tensor& norm_weight,
                                         const Tensor& gate_up_weight,
                                         const Tensor& down_weight,
                                         float eps = 1e-6f);

} // namespace motifcl
