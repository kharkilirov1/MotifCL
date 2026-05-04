#pragma once

#include <motifcl/train/optimizer.hpp>

namespace motifcl::optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<nn::Parameter*> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;

private:
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    float lr_ = 1e-3f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    int step_count_ = 0;
};

} // namespace motifcl::optim
