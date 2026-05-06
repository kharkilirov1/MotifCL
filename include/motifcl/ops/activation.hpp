#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor relu(const Tensor& x);
Tensor relu_backward_op(const Tensor& x, const Tensor& grad_out);
Tensor gelu(const Tensor& x);
Tensor gelu_backward_op(const Tensor& x, const Tensor& grad_out);
Tensor silu(const Tensor& x);
Tensor swiglu(const Tensor& packed);
Tensor swiglu_backward_op(const Tensor& packed, const Tensor& grad_out);
Tensor exp(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor rsqrt(const Tensor& x);

} // namespace motifcl
