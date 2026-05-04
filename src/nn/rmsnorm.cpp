#include <motifcl/nn/rmsnorm.hpp>

#include <motifcl/ops/norm.hpp>

namespace motifcl::nn {

RMSNorm::RMSNorm(Backend& backend, int features, float eps_value)
    : weight(Tensor::ones(backend, {features})), eps(eps_value) {}

Tensor RMSNorm::forward(const Tensor& x) { return motifcl::rmsnorm(x, weight.data, eps); }

} // namespace motifcl::nn
