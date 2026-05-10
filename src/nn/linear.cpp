#include <motifcl/nn/linear.hpp>

#include <cmath>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/quant.hpp>

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
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "Linear expects rank-2 f32 input");
    MCL_CHECK(x.shape()[1] == in_features_, "Linear input feature mismatch");
    if (quantized_weight_.valid() && !autograd::is_enabled()) {
        auto xq = quantize_q8_symmetric_rows(x);
        auto y = matmul(xq, quantized_weight_);
        return use_bias_ ? add_bias_rows(y, bias.data) : y;
    }
    auto y = matmul(x, weight.data);
    return use_bias_ ? add_bias_rows(y, bias.data) : y;
}

std::vector<Parameter*> Linear::parameters() {
    if (use_bias_) return {&weight, &bias};
    return {&weight};
}

void Linear::enable_quantized_inference(DType qdtype) {
    MCL_CHECK(qdtype == DType::Q8_0 || qdtype == DType::Q4_0,
              "Linear quantized inference expects qdtype Q8_0 or Q4_0");
    MCL_CHECK(weight.data.valid() && weight.data.dtype() == DType::F32 && weight.data.ndim() == 2,
              "Linear quantized inference expects rank-2 f32 weight");
    quantized_weight_dtype_ = qdtype;
    quantized_weight_ = qdtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(weight.data)
        : quantize_q8_symmetric_cols(weight.data);
}

void Linear::set_quantized_weight(const Tensor& qweight) {
    MCL_CHECK(qweight.valid() && (qweight.dtype() == DType::Q8_0 || qweight.dtype() == DType::Q4_0),
              "Linear quantized weight must be Q8_0 or Q4_0");
    MCL_CHECK(qweight.ndim() == 2 && qweight.shape()[0] == in_features_ && qweight.shape()[1] == out_features_,
              "Linear quantized weight shape mismatch");
    quantized_weight_ = qweight;
    quantized_weight_dtype_ = qweight.dtype();
}

void Linear::disable_quantized_inference() {
    quantized_weight_ = Tensor{};
    quantized_weight_dtype_ = DType::F32;
}

} // namespace motifcl::nn
