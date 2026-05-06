#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

bool backend_supports_fp16(const Backend& backend);
Tensor cast_f32_to_f16(const Tensor& x);
Tensor cast_f16_to_f32(const Tensor& x);

} // namespace motifcl
