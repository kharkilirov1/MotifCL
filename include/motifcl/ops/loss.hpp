#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor mse_backward_op(const Tensor& pred, const Tensor& target, const Tensor& grad_out);
Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets);

// Fused linear head + softmax cross-entropy: returns the same scalar loss as
// softmax_cross_entropy(matmul(xf, weight), targets), with `weight` in the nn layout
// [in_features, classes]. The backward pass folds the cross-entropy gradient into the
// two head GEMMs instead of materialising the [rows, classes] dlogits buffer. The folded form reproduces the standalone cached
// CE backward element for element, so gradients are bit-identical.
Tensor head_ce_fused(const Tensor& xf, const Tensor& weight, const Tensor& targets);
Tensor softmax_cross_entropy_backward_op(const Tensor& logits, const Tensor& targets, const Tensor& grad_out);

} // namespace motifcl
