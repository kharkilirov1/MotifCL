#include <motifcl/train/sgd.hpp>

#include <motifcl/ops/optim.hpp>

namespace motifcl::optim {

void Optimizer::zero_grad() {
    for (auto* p : params_) {
        if (p) p->zero_grad();
    }
}

SGD::SGD(std::vector<nn::Parameter*> params, float lr) : Optimizer(std::move(params)), lr_(lr) {}

void SGD::step() {
    for (auto* p : params_) {
        if (!p || !p->trainable || !p->data.grad()) continue;
        sgd_update(p->data, *p->data.grad(), lr_);
    }
}

} // namespace motifcl::optim
