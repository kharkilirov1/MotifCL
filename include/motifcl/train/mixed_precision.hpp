#pragma once

#include <motifcl/train/optimizer.hpp>

namespace motifcl::optim {

struct LossScaleUpdate {
    bool found_inf = false;
    float scale = 1.0f;
};

class DynamicLossScaler {
public:
    explicit DynamicLossScaler(float initial_scale = 65536.0f,
                               float growth_factor = 2.0f,
                               float backoff_factor = 0.5f,
                               int growth_interval = 2000);

    Tensor scale_loss(const Tensor& loss) const;
    void unscale_(const std::vector<nn::Parameter*>& params) const;
    bool has_overflow(const std::vector<nn::Parameter*>& params) const;
    LossScaleUpdate update(bool found_inf);

    float scale() const { return scale_; }
    void set_scale(float scale);
    int growth_tracker() const { return growth_tracker_; }

private:
    float scale_ = 65536.0f;
    float growth_factor_ = 2.0f;
    float backoff_factor_ = 0.5f;
    int growth_interval_ = 2000;
    int growth_tracker_ = 0;
};

class MixedPrecisionAdam : public Optimizer {
public:
    MixedPrecisionAdam(std::vector<nn::Parameter*> params,
                       float lr = 1e-3f,
                       float beta1 = 0.9f,
                       float beta2 = 0.999f,
                       float eps = 1e-8f,
                       float weight_decay = 0.0f);

    void step() override;
    bool step_scaled(DynamicLossScaler& scaler);
    void set_lr(float lr) override { lr_ = lr; }
    float lr() const override { return lr_; }
    void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }
    float weight_decay() const { return weight_decay_; }
    int step_count() const { return step_count_; }
    const std::vector<Tensor>& master_weights() const { return master_; }
    void refresh_master_from_params();

private:
    Tensor grad_as_f32(const Tensor& grad) const;
    Tensor param_as_f32(const nn::Parameter& param) const;
    void sync_param_from_master(std::size_t index);

    std::vector<Tensor> master_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    std::vector<DType> param_dtypes_;
    float lr_ = 1e-3f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    float weight_decay_ = 0.0f;
    int step_count_ = 0;
};

} // namespace motifcl::optim
