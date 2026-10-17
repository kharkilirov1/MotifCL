#include <motifcl/nn/moe7.hpp>

#include <algorithm>
#include <cmath>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/fog.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/tensor/tensor.hpp>

namespace motifcl::nn {

namespace {

// Distinct, stable seed per counter layer: the counter RNG stream advances once
// per update, so a deterministic seed keeps resume reproducible.
std::uint32_t seed_for(std::uint32_t base, int index) {
    return base + 7919u * static_cast<std::uint32_t>(index + 1);
}

} // namespace

MoE7FFN::MoE7FFN(Backend& backend, const MoE7Config& config) : config_(config) {
    MCL_CHECK(config_.dim > 0, "MoE7FFN requires dim > 0");
    MCL_CHECK(config_.hidden_expert > 0, "MoE7FFN requires hidden_expert > 0");
    MCL_CHECK(config_.hidden_shared > 0, "MoE7FFN requires hidden_shared > 0");

    const float init = 1.0f / std::sqrt(static_cast<float>(config_.dim));
    router_weight = Parameter(Tensor::randn(backend, {config_.dim, kExperts}, init), true);
    {
        std::vector<float> zeros(static_cast<std::size_t>(kExperts), 0.0f);
        router_bias = Tensor::from_cpu(backend, {kExperts}, DType::F32, zeros.data());
        router_bias.set_requires_grad(false);
    }

    int index = 0;
    for (auto& expert : experts_) {
        expert.up = std::make_unique<CounterStateLinear>(
            backend, config_.dim, 2 * config_.hidden_expert, config_.C, config_.counter_lr,
            0.0f, 1.0f, config_.rms_beta, config_.rms_eps, seed_for(config_.seed, index++));
        expert.down = std::make_unique<CounterStateLinear>(
            backend, config_.hidden_expert, config_.dim, config_.C, config_.counter_lr,
            0.0f, 1.0f, config_.rms_beta, config_.rms_eps, seed_for(config_.seed, index++));
    }
    shared_.up = std::make_unique<CounterStateLinear>(
        backend, config_.dim, 2 * config_.hidden_shared, config_.C, config_.counter_lr,
        0.0f, 1.0f, config_.rms_beta, config_.rms_eps, seed_for(config_.seed, index++));
    shared_.down = std::make_unique<CounterStateLinear>(
        backend, config_.hidden_shared, config_.dim, config_.C, config_.counter_lr,
        0.0f, 1.0f, config_.rms_beta, config_.rms_eps, seed_for(config_.seed, index++));
    last_load_.fill(0.0f);
}

Tensor MoE7FFN::run_expert(Expert& expert, const Tensor& x) {
    Tensor hidden = expert.up->forward(x);
    return expert.down->forward(swiglu(hidden));
}

std::vector<CounterStateLinear*> MoE7FFN::counter_layers() {
    std::vector<CounterStateLinear*> layers;
    layers.reserve(2 * static_cast<std::size_t>(kExperts) + 2);
    for (auto& expert : experts_) {
        layers.push_back(expert.up.get());
        layers.push_back(expert.down.get());
    }
    layers.push_back(shared_.up.get());
    layers.push_back(shared_.down.get());
    return layers;
}

void MoE7FFN::set_lr(float lr) {
    for (auto* layer : counter_layers()) layer->set_lr(lr);
}

void MoE7FFN::set_training(bool training) {
    for (auto* layer : counter_layers()) layer->set_training(training);
}

Tensor MoE7FFN::forward(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "MoE7FFN expects an f32 rank-2 input");
    MCL_CHECK(x.shape()[1] == config_.dim, "MoE7FFN input width mismatch");

    // Router logits; the bias shifts selection only (selection is a hard argmax,
    // the router gradient comes from the softmax over these same logits).
    Tensor logits = matmul(x, router_weight.data);
    Tensor biased = add_bias_rows(logits, router_bias);

    std::array<Tensor, kExperts> candidates;
    for (int e = 0; e < kExperts; ++e) {
        candidates[static_cast<std::size_t>(e)] = run_expert(experts_[static_cast<std::size_t>(e)], x);
    }
    Tensor routed = fog_hard_route7(biased, candidates);
    Tensor shared_out = run_expert(shared_, x);

    last_biased_logits_ = biased;
    last_input_ = x;
    last_rows_ = x.shape()[0];
    return add(routed, shared_out);
}

bool MoE7FFN::read_token_counts() {
    last_load_.fill(0.0f);
    last_logit_spread_ = 0.0f;
    last_logit_gap_ = 0.0f;
    if (!last_biased_logits_.valid() || last_rows_ <= 0) return false;
    const auto host = last_biased_logits_.to_vector<float>();
    const auto expected = static_cast<std::size_t>(last_rows_ * kExperts);
    MCL_CHECK(host.size() == expected, "MoE7FFN router logits size mismatch on read-back");
    last_logits_host_ = host;

    std::array<std::int64_t, kExperts> counts{};
    last_picks_.assign(static_cast<std::size_t>(last_rows_), 0);
    double spread_sum = 0.0;
    double gap_sum = 0.0;
    for (std::int64_t row = 0; row < last_rows_; ++row) {
        const auto base = static_cast<std::size_t>(row * kExperts);
        int best = 0;
        int second = 1;
        float best_value = host[base];
        float second_value = host[base + 1];
        float min_value = best_value;
        if (second_value > best_value) {
            std::swap(best, second);
            std::swap(best_value, second_value);
        }
        for (int k = 2; k < kExperts; ++k) {
            const float value = host[base + static_cast<std::size_t>(k)];
            if (value > best_value) {
                second = best;
                second_value = best_value;
                best = k;
                best_value = value;
            } else if (value > second_value) {
                second = k;
                second_value = value;
            }
            min_value = std::min(min_value, value);
        }
        min_value = std::min(min_value, second_value);
        spread_sum += (double)(best_value - min_value);
        gap_sum += (double)(best_value - second_value);
        counts[static_cast<std::size_t>(best)] += 1;
        last_picks_[static_cast<std::size_t>(row)] = best;
    }
    const float inv_rows = 1.0f / static_cast<float>(std::max<std::int64_t>(last_rows_, 1));
    last_logit_spread_ = (float)(spread_sum * inv_rows);
    last_logit_gap_ = (float)(gap_sum * inv_rows);
    for (int k = 0; k < kExperts; ++k) {
        last_load_[static_cast<std::size_t>(k)] =
            static_cast<float>(counts[static_cast<std::size_t>(k)]) * inv_rows;
    }
    return true;
}

void MoE7FFN::prepare_balance_gradient(float aux_weight) {
    balance_grad_ = Tensor();
    if (!(aux_weight > 0.0f) || last_rows_ <= 0) return;
    if (last_logits_host_.size() != static_cast<std::size_t>(last_rows_ * kExperts)) return;

    // p = softmax(logits) per token; dl[n,j] = (E/N) * p[n,j] * (f_j - sum_e f_e p[n,e]).
    // This is the exact gradient of L_aux = E * sum_e f_e * P_e with f held constant.
    std::vector<float> dl((std::size_t)(last_rows_ * kExperts), 0.0f);
    std::array<float, kExperts> p{};
    const float e_over_n = static_cast<float>(kExperts) / static_cast<float>(last_rows_);
    for (std::int64_t n = 0; n < last_rows_; ++n) {
        const auto base = static_cast<std::size_t>(n * kExperts);
        float mx = last_logits_host_[base];
        for (int k = 1; k < kExperts; ++k) mx = std::max(mx, last_logits_host_[base + static_cast<std::size_t>(k)]);
        float z = 0.0f;
        for (int k = 0; k < kExperts; ++k) {
            p[static_cast<std::size_t>(k)] = std::exp(last_logits_host_[base + static_cast<std::size_t>(k)] - mx);
            z += p[static_cast<std::size_t>(k)];
        }
        float load_dot = 0.0f;
        for (int k = 0; k < kExperts; ++k) {
            p[static_cast<std::size_t>(k)] /= z;
            load_dot += last_load_[static_cast<std::size_t>(k)] * p[static_cast<std::size_t>(k)];
        }
        for (int j = 0; j < kExperts; ++j) {
            const float pj = p[static_cast<std::size_t>(j)];
            dl[base + static_cast<std::size_t>(j)] =
                e_over_n * aux_weight * pj * (last_load_[static_cast<std::size_t>(j)] - load_dot);
        }
    }
    balance_grad_ = Tensor::from_cpu(router_weight.data.backend(), {last_rows_, kExperts},
                                     DType::F32, dl.data());
}

void MoE7FFN::inject_balance_gradient() {
    if (!balance_grad_.valid() || !last_input_.valid()) return;
    // gW = x^T * (dL_aux/dlogits) -> [dim, kExperts], accumulated into the router grad.
    Tensor gW = matmul_transpose_a(last_input_, balance_grad_);
    router_weight.data.backward(gW);
}

void MoE7FFN::update_balance_bias(float gamma) {
    if (!(gamma > 0.0f)) return;
    std::vector<float> host = router_bias.to_vector<float>();
    MCL_CHECK(host.size() == static_cast<std::size_t>(kExperts), "MoE7FFN router bias size mismatch");
    const float target = 1.0f / static_cast<float>(kExperts);
    for (int k = 0; k < kExperts; ++k) {
        const auto index = static_cast<std::size_t>(k);
        // Sign update (DeepSeek-v3 style): a starved expert gains at a fixed rate
        // per step instead of proportionally to its deficit, so it can re-enter
        // the top-1 argmax within ~1/gamma steps even after a large gap.
        const float delta = target - last_load_[index];
        const float step = (delta > 0.0f) ? 1.0f : ((delta < 0.0f) ? -1.0f : 0.0f);
        host[index] += gamma * step;
        host[index] = std::max(-config_.balance_clamp, std::min(config_.balance_clamp, host[index]));
    }
    Tensor updated = Tensor::from_cpu(router_bias.backend(), {kExperts}, DType::F32, host.data());
    updated.set_requires_grad(false);
    router_bias = std::move(updated);
}

} // namespace motifcl::nn
