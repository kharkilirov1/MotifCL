#include <motifcl/nn/rmsnorm.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/norm.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cmath>
#include <vector>

namespace motifcl::nn {

RMSNorm::RMSNorm(Backend& backend, int features, float eps_value, bool layer_norm)
    : weight(Tensor::ones(backend, {features})),
      bias(Tensor::zeros(backend, {features})),
      eps(eps_value),
      layer_norm_(layer_norm) {}

Tensor RMSNorm::forward(const Tensor& x) {
    if (!layer_norm_) return motifcl::rmsnorm(x, weight.data, eps);
    if (!x.backend().is_vulkan()) return motifcl::layernorm(x, weight.data, bias.data, eps);

    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2,
              "Vulkan LayerNorm reference expects f32 [rows, cols]");
    const int64_t rows = x.shape()[0];
    const int64_t cols = x.shape()[1];
    MCL_CHECK(weight.data.shape() == Shape({cols}) && bias.data.shape() == Shape({cols}),
              "Vulkan LayerNorm reference parameter shape mismatch");
    const auto input = x.to_vector<float>();
    const auto weights = weight.data.to_vector<float>();
    const auto biases = bias.data.to_vector<float>();
    std::vector<float> output(input.size());
    for (int64_t row = 0; row < rows; ++row) {
        double mean = 0.0;
        for (int64_t col = 0; col < cols; ++col) {
            mean += input[static_cast<std::size_t>(row * cols + col)];
        }
        mean /= static_cast<double>(cols);
        double variance = 0.0;
        for (int64_t col = 0; col < cols; ++col) {
            const double centered =
                static_cast<double>(input[static_cast<std::size_t>(row * cols + col)]) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(cols);
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(variance) + eps);
        for (int64_t col = 0; col < cols; ++col) {
            const auto index = static_cast<std::size_t>(row * cols + col);
            output[index] =
                (input[index] - static_cast<float>(mean)) * inv_std *
                    weights[static_cast<std::size_t>(col)] +
                biases[static_cast<std::size_t>(col)];
        }
    }
    return Tensor::from_cpu(x.backend(), x.shape(), DType::F32, output.data());
}

std::vector<Parameter*> RMSNorm::parameters() {
    return layer_norm_ ? std::vector<Parameter*>{&weight, &bias}
                       : std::vector<Parameter*>{&weight};
}

} // namespace motifcl::nn
