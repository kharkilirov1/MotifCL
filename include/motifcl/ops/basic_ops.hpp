#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor fill(Backend& backend, const Shape& shape, float value);
Tensor add(const Tensor& a, const Tensor& b);
Tensor add_broadcast(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul_broadcast(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor add_scalar(const Tensor& x, float value);
Tensor mul_scalar(const Tensor& x, float value);
Tensor scale(const Tensor& x, float alpha);
void scale_inplace(Tensor& x, float alpha);
void add_inplace(Tensor& dst, const Tensor& src);
Tensor add_bias_rows(const Tensor& x, const Tensor& bias);
Tensor add_bias_gelu_rows(const Tensor& x, const Tensor& bias);
Tensor zeros_like(const Tensor& x);
Tensor ones_like(const Tensor& x);

} // namespace motifcl
