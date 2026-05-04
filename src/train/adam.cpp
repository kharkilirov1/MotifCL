#include <motifcl/train/adam.hpp>

#include <motifcl/ops/optim.hpp>

namespace motifcl::optim {

Adam::Adam(std::vector<nn::Parameter*> params, float lr, float beta1, float beta2, float eps)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (auto* p : params_) {
        m_.push_back(Tensor::zeros(p->data.backend(), p->data.shape()));
        v_.push_back(Tensor::zeros(p->data.backend(), p->data.shape()));
    }
}

void Adam::step() {
    ++step_count_;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p || !p->trainable || !p->data.grad()) continue;
        adam_update(p->data, *p->data.grad(), m_[i], v_[i], lr_, beta1_, beta2_, eps_, step_count_);
    }
}

} // namespace motifcl::optim
