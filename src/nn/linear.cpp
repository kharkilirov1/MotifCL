#include <motifcl/nn/linear.hpp>

#include <cmath>

#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::nn {

Linear::Linear(Backend& backend, int in_features, int out_features, bool use_bias)
    : weight(Tensor::randn(backend, {in_features, out_features}, 1.0f / std::sqrt(static_cast<float>(in_features)))),
      bias(Tensor::zeros(backend, {out_features})),
      in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias) {
    weight.data.set_requires_grad(true);
    bias.data.set_requires_grad(use_bias_);
}

Tensor Linear::forward(const Tensor& x) {
    auto y = matmul(x, weight.data);
    return use_bias_ ? add_bias_rows(y, bias.data) : y;
}

std::vector<Parameter*> Linear::parameters() {
    if (use_bias_) return {&weight, &bias};
    return {&weight};
}

} // namespace motifcl::nn
