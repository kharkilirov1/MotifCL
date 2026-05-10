#pragma once

#include <motifcl/train/optimizer.hpp>

namespace motifcl::optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<nn::Parameter*> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void set_lr(float lr) override { lr_ = lr; }
    float lr() const override { return lr_; }
    void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }
    float weight_decay() const { return weight_decay_; }

private:
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    Tensor state_;
    float lr_ = 1e-3f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    float weight_decay_ = 0.0f;
    int step_count_ = 0;
};

} // namespace motifcl::optim
