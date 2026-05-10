#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

void sgd_update(Tensor& param, const Tensor& grad, float lr);
void adam_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                 float lr, float beta1, float beta2, float eps, int step);
void adam_update_fast(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                      float lr, float beta1, float beta2, float eps, int step, float weight_decay = 0.0f);

} // namespace motifcl
