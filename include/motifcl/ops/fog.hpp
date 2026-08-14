#pragma once

#include <array>
#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

// Pairwise complex product over adjacent coordinates followed by row-wise L2
// normalization to sqrt(d_model). This is the law-preserving BLOCK_PRODUCT
// primitive used by FOG register_machine_v3.
Tensor fog_block_product(const Tensor& value, const Tensor& addressed);
Tensor fog_block_product_backward_value(const Tensor& value,
                                        const Tensor& addressed,
                                        const Tensor& grad_out);
Tensor fog_block_product_backward_addressed(const Tensor& value,
                                            const Tensor& addressed,
                                            const Tensor& grad_out);

// Seven-way hard routing with a straight-through softmax gradient. Candidate
// tensors are [rows,d_model], logits are [rows,7].
Tensor fog_hard_route7(const Tensor& logits, const std::array<Tensor, 7>& candidates);

} // namespace motifcl
