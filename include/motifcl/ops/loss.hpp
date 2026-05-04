#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor mse_backward_op(const Tensor& pred, const Tensor& target, const Tensor& grad_out);
Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor softmax_cross_entropy_backward_op(const Tensor& logits, const Tensor& targets, const Tensor& grad_out);

} // namespace motifcl
