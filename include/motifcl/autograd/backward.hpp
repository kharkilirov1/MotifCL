#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl::autograd {

void backward(Tensor loss);

} // namespace motifcl::autograd
