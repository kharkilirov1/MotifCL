#include <motifcl/train/adam.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/optim.hpp>
#include <motifcl/runtime/backend.hpp>

#include <algorithm>
#include <cmath>

namespace motifcl::optim {

namespace {

std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

struct AdamUpdateRef {
    Tensor* param = nullptr;
    Tensor* grad = nullptr;
    Tensor* m = nullptr;
    Tensor* v = nullptr;
};

void adam_update_fast4(const std::vector<AdamUpdateRef>& group,
                       float lr,
                       float beta1,
                       float beta2,
                       float eps,
                       float weight_decay,
                       int step) {
    MCL_CHECK(!group.empty() && group.size() <= 4, "adam_update_fast4 expects 1..4 tensors");
    auto& backend = group.front().param->backend();
    int max_n = 0;
    for (const auto& item : group) {
        MCL_CHECK(item.param && item.grad && item.m && item.v, "adam_update_fast4 null tensor");
        MCL_CHECK(item.param->backend_ptr() == &backend && item.grad->backend_ptr() == &backend &&
                      item.m->backend_ptr() == &backend && item.v->backend_ptr() == &backend,
                  "adam_update_fast4 tensors must share backend");
        MCL_CHECK(item.param->dtype() == DType::F32 && item.grad->dtype() == DType::F32 &&
                      item.m->dtype() == DType::F32 && item.v->dtype() == DType::F32,
                  "adam_update_fast4 supports f32 only");
        MCL_CHECK(item.param->shape() == item.grad->shape() && item.param->shape() == item.m->shape() &&
                      item.param->shape() == item.v->shape(),
                  "adam_update_fast4 shape mismatch");
        max_n = std::max(max_n, static_cast<int>(item.param->numel()));
    }
    MCL_CHECK(max_n > 0, "adam_update_fast4 empty tensor");

    auto k = backend.kernels.get("adam_update_f32_fast4");
    int arg = 0;
    std::vector<int> inputs;
    std::vector<int> outputs;
    inputs.reserve(group.size() * 4);
    outputs.reserve(group.size() * 3);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        const auto& item = slot < group.size() ? group[slot] : group.front();
        const int n = slot < group.size() ? static_cast<int>(item.param->numel()) : 0;
        k.set_arg(arg++, item.param->buffer());
        k.set_arg(arg++, item.grad->buffer());
        k.set_arg(arg++, item.m->buffer());
        k.set_arg(arg++, item.v->buffer());
        k.set_arg(arg++, n);
        if (slot < group.size()) {
            inputs.push_back(item.param->id());
            inputs.push_back(item.grad->id());
            inputs.push_back(item.m->id());
            inputs.push_back(item.v->id());
            outputs.push_back(item.param->id());
            outputs.push_back(item.m->id());
            outputs.push_back(item.v->id());
        }
    }
    const float inv_bias1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
    const float inv_bias2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
    k.set_arg(arg++, lr);
    k.set_arg(arg++, beta1);
    k.set_arg(arg++, beta2);
    k.set_arg(arg++, eps);
    k.set_arg(arg++, inv_bias1);
    k.set_arg(arg++, inv_bias2);
    k.set_arg(arg++, weight_decay);
    k.launch1d(round_up(static_cast<std::size_t>(max_n), 256), 256);
    autograd::record_op("adam_update_f32_fast4", std::move(inputs), std::move(outputs));
}

void adam_update_state(Tensor& state, float beta1, float beta2) {
    MCL_CHECK(state.valid() && state.dtype() == DType::F32 && state.numel() >= 3,
              "adam_update_state expects f32 state[3]");
    auto k = state.backend().kernels.get("adam_update_state_f32");
    k.set_arg(0, state.buffer());
    k.set_arg(1, beta1);
    k.set_arg(2, beta2);
    k.launch1d(1, 1);
    autograd::record_op("adam_update_state_f32", {state.id()}, {state.id()});
}

void adam_update_fast8_state(const std::vector<AdamUpdateRef>& group,
                             Tensor& state,
                             float lr,
                             float beta1,
                             float beta2,
                             float eps,
                             float weight_decay) {
    MCL_CHECK(!group.empty() && group.size() <= 8, "adam_update_fast8_state expects 1..8 tensors");
    auto& backend = group.front().param->backend();
    MCL_CHECK(state.valid() && state.backend_ptr() == &backend, "adam_update_fast8_state state backend mismatch");
    int max_n = 0;
    for (const auto& item : group) {
        MCL_CHECK(item.param && item.grad && item.m && item.v, "adam_update_fast8_state null tensor");
        MCL_CHECK(item.param->backend_ptr() == &backend && item.grad->backend_ptr() == &backend &&
                      item.m->backend_ptr() == &backend && item.v->backend_ptr() == &backend,
                  "adam_update_fast8_state tensors must share backend");
        MCL_CHECK(item.param->dtype() == DType::F32 && item.grad->dtype() == DType::F32 &&
                      item.m->dtype() == DType::F32 && item.v->dtype() == DType::F32,
                  "adam_update_fast8_state supports f32 only");
        MCL_CHECK(item.param->shape() == item.grad->shape() && item.param->shape() == item.m->shape() &&
                      item.param->shape() == item.v->shape(),
                  "adam_update_fast8_state shape mismatch");
        max_n = std::max(max_n, static_cast<int>(item.param->numel()));
    }
    MCL_CHECK(max_n > 0, "adam_update_fast8_state empty tensor");

    auto k = backend.kernels.get("adam_update_f32_fast8_state");
    int arg = 0;
    std::vector<int> inputs{state.id()};
    std::vector<int> outputs;
    inputs.reserve(1 + group.size() * 4);
    outputs.reserve(group.size() * 3);
    for (std::size_t slot = 0; slot < 8; ++slot) {
        const auto& item = slot < group.size() ? group[slot] : group.front();
        const int n = slot < group.size() ? static_cast<int>(item.param->numel()) : 0;
        k.set_arg(arg++, item.param->buffer());
        k.set_arg(arg++, item.grad->buffer());
        k.set_arg(arg++, item.m->buffer());
        k.set_arg(arg++, item.v->buffer());
        k.set_arg(arg++, n);
        if (slot < group.size()) {
            inputs.push_back(item.param->id());
            inputs.push_back(item.grad->id());
            inputs.push_back(item.m->id());
            inputs.push_back(item.v->id());
            outputs.push_back(item.param->id());
            outputs.push_back(item.m->id());
            outputs.push_back(item.v->id());
        }
    }
    k.set_arg(arg++, state.buffer());
    k.set_arg(arg++, lr);
    k.set_arg(arg++, beta1);
    k.set_arg(arg++, beta2);
    k.set_arg(arg++, eps);
    k.set_arg(arg++, weight_decay);
    k.launch1d(round_up(static_cast<std::size_t>(max_n), 256), 256);
    autograd::record_op("adam_update_f32_fast8_state", std::move(inputs), std::move(outputs));
}

} // namespace

Adam::Adam(std::vector<nn::Parameter*> params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {
    Backend* backend = nullptr;
    for (auto* p : params_) {
        MCL_CHECK(p != nullptr, "Adam received null parameter");
        if (!backend) backend = p->data.backend_ptr();
        MCL_CHECK(p->data.backend_ptr() == backend, "Adam currently expects all parameters on one backend");
        m_.push_back(Tensor::zeros(p->data.backend(), p->data.shape()));
        v_.push_back(Tensor::zeros(p->data.backend(), p->data.shape()));
    }
    if (backend) state_ = Tensor::zeros(*backend, {3}, DType::F32);
}

void Adam::step() {
    ++step_count_;
    std::vector<AdamUpdateRef> group;
    group.reserve(8);
    if (state_.valid()) {
        adam_update_state(state_, beta1_, beta2_);
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (!p || !p->trainable || !p->data.grad()) continue;
        AdamUpdateRef item{&p->data, p->data.grad().get(), &m_[i], &v_[i]};
        if (!group.empty() && item.param->backend_ptr() != group.front().param->backend_ptr()) {
            if (state_.valid()) adam_update_fast8_state(group, state_, lr_, beta1_, beta2_, eps_, weight_decay_);
            else adam_update_fast4(group, lr_, beta1_, beta2_, eps_, weight_decay_, step_count_);
            group.clear();
        }
        group.push_back(item);
        if (group.size() == 8) {
            if (state_.valid()) adam_update_fast8_state(group, state_, lr_, beta1_, beta2_, eps_, weight_decay_);
            else adam_update_fast4(group, lr_, beta1_, beta2_, eps_, weight_decay_, step_count_);
            group.clear();
        }
    }
    if (!group.empty()) {
        if (state_.valid()) adam_update_fast8_state(group, state_, lr_, beta1_, beta2_, eps_, weight_decay_);
        else adam_update_fast4(group, lr_, beta1_, beta2_, eps_, weight_decay_, step_count_);
    }
}

} // namespace motifcl::optim
