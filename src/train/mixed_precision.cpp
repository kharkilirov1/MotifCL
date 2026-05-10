#include <motifcl/train/mixed_precision.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/fp16.hpp>
#include <motifcl/ops/optim.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace motifcl::optim {

namespace {

void require_valid_scale(float scale, const char* op) {
    MCL_CHECK(std::isfinite(scale) && scale > 0.0f, std::string(op) + " scale must be positive and finite");
}

bool tensor_has_nonfinite_f32(const Tensor& tensor) {
    const auto values = tensor.to_vector<float>();
    for (float value : values) {
        if (!std::isfinite(value)) return true;
    }
    return false;
}

} // namespace

DynamicLossScaler::DynamicLossScaler(float initial_scale,
                                     float growth_factor,
                                     float backoff_factor,
                                     int growth_interval)
    : scale_(initial_scale),
      growth_factor_(growth_factor),
      backoff_factor_(backoff_factor),
      growth_interval_(growth_interval) {
    require_valid_scale(scale_, "DynamicLossScaler");
    MCL_CHECK(std::isfinite(growth_factor_) && growth_factor_ > 1.0f, "DynamicLossScaler growth_factor must be > 1");
    MCL_CHECK(std::isfinite(backoff_factor_) && backoff_factor_ > 0.0f && backoff_factor_ < 1.0f,
              "DynamicLossScaler backoff_factor must be in (0, 1)");
    MCL_CHECK(growth_interval_ > 0, "DynamicLossScaler growth_interval must be positive");
}

Tensor DynamicLossScaler::scale_loss(const Tensor& loss) const {
    MCL_CHECK(loss.dtype() == DType::F32, "DynamicLossScaler::scale_loss expects f32 loss");
    return motifcl::scale(loss, scale_);
}

void DynamicLossScaler::unscale_(const std::vector<nn::Parameter*>& params) const {
    const float inv_scale = 1.0f / scale_;
    for (auto* param : params) {
        if (!param || !param->trainable || !param->data.grad()) continue;
        Tensor& grad = *param->data.grad();
        if (grad.dtype() == DType::F32) {
            motifcl::scale_inplace(grad, inv_scale);
        } else if (grad.dtype() == DType::F16) {
            auto g32 = motifcl::cast_f16_to_f32(grad);
            motifcl::scale_inplace(g32, inv_scale);
            param->data._set_grad(g32);
        } else {
            MCL_CHECK(false, "DynamicLossScaler::unscale_ supports f32/f16 gradients only");
        }
    }
}

bool DynamicLossScaler::has_overflow(const std::vector<nn::Parameter*>& params) const {
    for (auto* param : params) {
        if (!param || !param->trainable || !param->data.grad()) continue;
        const Tensor& grad = *param->data.grad();
        if (grad.dtype() == DType::F32) {
            if (tensor_has_nonfinite_f32(grad)) return true;
        } else if (grad.dtype() == DType::F16) {
            if (tensor_has_nonfinite_f32(motifcl::cast_f16_to_f32(grad))) return true;
        } else {
            return true;
        }
    }
    return false;
}

LossScaleUpdate DynamicLossScaler::update(bool found_inf) {
    if (found_inf) {
        scale_ = std::max(1.0f, scale_ * backoff_factor_);
        growth_tracker_ = 0;
        return {true, scale_};
    }
    ++growth_tracker_;
    if (growth_tracker_ >= growth_interval_) {
        scale_ *= growth_factor_;
        if (!std::isfinite(scale_)) scale_ = std::numeric_limits<float>::max();
        growth_tracker_ = 0;
    }
    return {false, scale_};
}

void DynamicLossScaler::set_scale(float scale) {
    require_valid_scale(scale, "DynamicLossScaler::set_scale");
    scale_ = scale;
    growth_tracker_ = 0;
}

MixedPrecisionAdam::MixedPrecisionAdam(std::vector<nn::Parameter*> params,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float eps,
                                       float weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay) {
    MCL_CHECK(std::isfinite(lr_) && lr_ >= 0.0f, "MixedPrecisionAdam lr must be finite and non-negative");
    MCL_CHECK(beta1_ >= 0.0f && beta1_ < 1.0f && beta2_ >= 0.0f && beta2_ < 1.0f, "MixedPrecisionAdam betas must be in [0, 1)");
    MCL_CHECK(std::isfinite(eps_) && eps_ > 0.0f, "MixedPrecisionAdam eps must be positive");
    refresh_master_from_params();
}

Tensor MixedPrecisionAdam::param_as_f32(const nn::Parameter& param) const {
    MCL_CHECK(param.data.valid(), "MixedPrecisionAdam parameter tensor is invalid");
    if (param.data.dtype() == DType::F32) {
        auto host = param.data.to_vector<float>();
        return Tensor::from_cpu(param.data.backend(), param.data.shape(), DType::F32, host.data());
    }
    if (param.data.dtype() == DType::F16) {
        return motifcl::cast_f16_to_f32(param.data);
    }
    MCL_CHECK(false, "MixedPrecisionAdam supports f32/f16 parameters only");
    return {};
}

Tensor MixedPrecisionAdam::grad_as_f32(const Tensor& grad) const {
    if (grad.dtype() == DType::F32) return grad;
    if (grad.dtype() == DType::F16) return motifcl::cast_f16_to_f32(grad);
    MCL_CHECK(false, "MixedPrecisionAdam supports f32/f16 gradients only");
    return {};
}

void MixedPrecisionAdam::refresh_master_from_params() {
    master_.clear();
    m_.clear();
    v_.clear();
    param_dtypes_.clear();
    master_.reserve(params_.size());
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    param_dtypes_.reserve(params_.size());
    for (auto* param : params_) {
        MCL_CHECK(param != nullptr, "MixedPrecisionAdam received null parameter");
        param_dtypes_.push_back(param->data.dtype());
        master_.push_back(param_as_f32(*param));
        m_.push_back(Tensor::zeros(param->data.backend(), param->data.shape(), DType::F32));
        v_.push_back(Tensor::zeros(param->data.backend(), param->data.shape(), DType::F32));
    }
}

void MixedPrecisionAdam::sync_param_from_master(std::size_t index) {
    auto* param = params_[index];
    if (!param || !param->trainable) return;
    if (param_dtypes_[index] == DType::F16) {
        param->data = motifcl::cast_f32_to_f16(master_[index]);
    } else if (param_dtypes_[index] == DType::F32) {
        param->data = master_[index];
    } else {
        MCL_CHECK(false, "MixedPrecisionAdam unsupported parameter dtype while syncing");
    }
    param->data.set_requires_grad(param->trainable);
}

void MixedPrecisionAdam::step() {
    ++step_count_;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* param = params_[i];
        if (!param || !param->trainable || !param->data.grad()) continue;
        auto grad = grad_as_f32(*param->data.grad());
        adam_update_fast(master_[i], grad, m_[i], v_[i], lr_, beta1_, beta2_, eps_, step_count_, weight_decay_);
        sync_param_from_master(i);
    }
}

bool MixedPrecisionAdam::step_scaled(DynamicLossScaler& scaler) {
    if (scaler.has_overflow(params_)) {
        scaler.update(true);
        return false;
    }
    scaler.unscale_(params_);
    step();
    scaler.update(false);
    return true;
}

} // namespace motifcl::optim
