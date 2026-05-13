#include <motifcl/nn/linear.hpp>

#include <cmath>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/quant.hpp>

namespace motifcl::nn {

Linear::Linear(Backend& backend, int in_features, int out_features, bool use_bias, bool skip_weight_init)
    : weight(),
      bias(Tensor::zeros(backend, {out_features})),
      in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias) {
    if (!skip_weight_init) {
        weight = Parameter(Tensor::randn(backend, {in_features, out_features},
                                         1.0f / std::sqrt(static_cast<float>(in_features))));
        weight.data.set_requires_grad(true);
    } else {
        weight.trainable = false;
    }
    bias.data.set_requires_grad(use_bias_);
}

Tensor Linear::forward(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "Linear expects rank-2 f32 input");
    MCL_CHECK(x.shape()[1] == in_features_, "Linear input feature mismatch");
    if (quantized_weight_.valid() && (!autograd::is_enabled() || !weight.data.valid())) {
        const bool m1 = x.shape()[0] == 1;
        const Tensor& decode_weight = decode_quantized_weight_.valid() ? decode_quantized_weight_ : quantized_weight_;
        const DType decode_dtype = decode_quantized_weight_.valid() ? decode_quantized_weight_dtype_ : quantized_weight_dtype_;
        if ((decode_dtype == DType::Q4_0 || decode_dtype == DType::Q4_0_COL ||
             decode_dtype == DType::Q4_K || decode_dtype == DType::Q5_K ||
             decode_dtype == DType::Q6_K) &&
            m1) {
            auto y = matmul(x, decode_weight);
            return use_bias_ ? add_bias_rows(y, bias.data) : y;
        }
        auto xq = quantize_q8_symmetric_rows(x);
        auto y = matmul(xq, quantized_weight_);
        return use_bias_ ? add_bias_rows(y, bias.data) : y;
    }
    MCL_CHECK(weight.data.valid(), "Linear dense weight is not initialized");
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
    decode_quantized_weight_ = Tensor{};
    decode_quantized_weight_dtype_ = DType::F32;
}

void Linear::set_quantized_weight(const Tensor& qweight) {
    MCL_CHECK(qweight.valid() && (qweight.dtype() == DType::Q8_0 || qweight.dtype() == DType::Q4_0 ||
                                  qweight.dtype() == DType::Q4_0_COL || qweight.dtype() == DType::Q4_K ||
                                  qweight.dtype() == DType::Q5_K || qweight.dtype() == DType::Q6_K),
              "Linear quantized weight must be Q8_0, Q4_0, Q4_0_COL, Q4_K, Q5_K, or Q6_K");
    MCL_CHECK(qweight.ndim() == 2 && qweight.shape()[0] == in_features_ && qweight.shape()[1] == out_features_,
              "Linear quantized weight shape mismatch");
    quantized_weight_ = qweight;
    quantized_weight_dtype_ = qweight.dtype();
    decode_quantized_weight_ = Tensor{};
    decode_quantized_weight_dtype_ = DType::F32;
}

void Linear::set_decode_quantized_weight(const Tensor& qweight) {
    MCL_CHECK(qweight.valid() && (qweight.dtype() == DType::Q8_0 || qweight.dtype() == DType::Q4_0 ||
                                  qweight.dtype() == DType::Q4_0_COL || qweight.dtype() == DType::Q4_K ||
                                  qweight.dtype() == DType::Q5_K || qweight.dtype() == DType::Q6_K),
              "Linear decode quantized weight must be Q8_0, Q4_0, Q4_0_COL, Q4_K, Q5_K, or Q6_K");
    MCL_CHECK(qweight.ndim() == 2 && qweight.shape()[0] == in_features_ && qweight.shape()[1] == out_features_,
              "Linear decode quantized weight shape mismatch");
    decode_quantized_weight_ = qweight;
    decode_quantized_weight_dtype_ = qweight.dtype();
}

void Linear::disable_quantized_inference() {
    quantized_weight_ = Tensor{};
    quantized_weight_dtype_ = DType::F32;
    decode_quantized_weight_ = Tensor{};
    decode_quantized_weight_dtype_ = DType::F32;
}

} // namespace motifcl::nn
