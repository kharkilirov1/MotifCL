#pragma once

#include <array>
#include <functional>
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

// Same op, with an explicit "this candidate received a zero gradient" hook.
// The hook returns true when it handled the candidate's side effects and its
// backward graph must be skipped; false runs the normal masked backward.
// The dense backward materialises an all-zero gradient tensor for every loser
// and runs its whole backward graph on it, which for a counter-state expert
// means a grad_w GEMM over the full weight matrix that can only produce zeros.
// When `on_zero_candidate` is set it is called with the candidate index instead,
// and the caller is responsible for applying whatever side effects a zero
// gradient would have had (see CounterStateLinear::apply_zero_grad_update).
// The router gradient is unchanged: it is still computed from every candidate's
// values, which is why the losers must still be evaluated in forward.
Tensor fog_hard_route7(const Tensor& logits, const std::array<Tensor, 7>& candidates,
                       std::function<bool(std::uint32_t)> on_zero_candidate);

} // namespace motifcl
