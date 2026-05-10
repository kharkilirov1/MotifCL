#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);
Tensor rmsnorm_backward_x(const Tensor& x, const Tensor& weight, const Tensor& grad_out, float eps = 1e-6f);
Tensor rmsnorm_backward_x_residual(const Tensor& x, const Tensor& weight, const Tensor& grad_out,
                                   const Tensor& residual_grad, float eps = 1e-6f);
Tensor rmsnorm_backward_weight(const Tensor& x, const Tensor& weight, const Tensor& grad_out, float eps = 1e-6f);
Tensor layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps = 1e-6f);

} // namespace motifcl
