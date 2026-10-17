// Reference test for nn::MoE7FFN on the Vulkan backend.
//
// 1. Forward equality: the routed MoE output is compared row-for-row against a
//    host reference built from each counter layer's decoded weight
//    (SwiGLU: gate * sigmoid(gate) * up; hard top-1 selection by host argmax).
// 2. Routing agreement: the module's own token counts (read back from its router
//    logits) must match the host argmax distribution exactly.
// 3. Gradient semantics: after a backward, only experts that actually won tokens
//    may have received gradient -- observed through the counter row-RMS `v`,
//    which stays zero under an all-zero gradient and becomes positive otherwise.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/moe7.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace {

double max_abs(const std::vector<float>& v) {
    double m = 0.0;
    for (float x : v) m = std::max(m, (double)std::fabs(x));
    return m;
}

std::vector<float> linear(const std::vector<float>& x, int rows, int in, const std::vector<float>& w, int out) {
    std::vector<float> y((std::size_t)rows * out, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out; ++o) {
            float s = 0.0f;
            for (int i = 0; i < in; ++i) s += w[(std::size_t)o * in + i] * x[(std::size_t)r * in + i];
            y[(std::size_t)r * out + o] = s;
        }
    }
    return y;
}

std::vector<float> swiglu(const std::vector<float>& packed, int rows, int hidden) {
    std::vector<float> act((std::size_t)rows * hidden, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < hidden; ++c) {
            const float gate = packed[(std::size_t)r * 2 * hidden + c];
            const float up = packed[(std::size_t)r * 2 * hidden + hidden + c];
            act[(std::size_t)r * hidden + c] = gate * (1.0f / (1.0f + std::exp(-gate))) * up;
        }
    }
    return act;
}

} // namespace

int main() {
    using namespace motifcl;
    try {
        const auto probe = probe_vulkan_runtime();
        if (!probe.available()) {
            std::printf("SKIP test_moe7_reference: no Vulkan device\n");
            return 0;
        }
        Backend backend = Backend::create_vulkan();

        const int dim = 32, h_exp = 16, h_shared = 16, rows = 8;
        nn::MoE7Config cfg;
        cfg.dim = dim;
        cfg.hidden_expert = h_exp;
        cfg.hidden_shared = h_shared;
        cfg.C = 11;
        cfg.counter_lr = 4e-3f;
        cfg.seed = 20261016u;
        auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);

        std::vector<float> x_host((std::size_t)rows * dim, 0.0f);
        std::mt19937 rng(11);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (auto& v : x_host) v = gauss(rng);

        // ---- 1. forward equality ----
        moe->set_training(false);
        auto x = Tensor::from_cpu(backend, {rows, dim}, DType::F32, x_host.data());
        auto y = moe->forward(x);
        const auto y_vk = y.to_vector<float>();

        std::vector<std::vector<float>> expert_out(nn::MoE7FFN::kExperts);
        for (int e = 0; e < nn::MoE7FFN::kExperts; ++e) {
            auto counter_layers = moe->counter_layers();
            auto& up = *counter_layers[(std::size_t)(2 * e)];
            auto& down = *counter_layers[(std::size_t)(2 * e + 1)];
            const auto w_up = up.decode_weight().to_vector<float>();
            const auto w_dn = down.decode_weight().to_vector<float>();
            const auto packed = linear(x_host, rows, dim, w_up, 2 * h_exp);
            const auto act = swiglu(packed, rows, h_exp);
            expert_out[(std::size_t)e] = linear(act, rows, h_exp, w_dn, dim);
        }
        auto counter_layers = moe->counter_layers();
        const auto w_sh_up = counter_layers[(std::size_t)(2 * nn::MoE7FFN::kExperts)]->decode_weight().to_vector<float>();
        const auto w_sh_dn = counter_layers[(std::size_t)(2 * nn::MoE7FFN::kExperts + 1)]->decode_weight().to_vector<float>();
        const auto shared_out = linear(swiglu(linear(x_host, rows, dim, w_sh_up, 2 * h_shared), rows, h_shared),
                                       rows, h_shared, w_sh_dn, dim);

        const auto w_r = moe->router_weight.data.to_vector<float>();      // [dim, 7]
        const auto bias = moe->router_bias.to_vector<float>();            // [7]
        std::vector<int> host_pick(rows, 0);
        std::vector<int> host_counts(nn::MoE7FFN::kExperts, 0);
        for (int r = 0; r < rows; ++r) {
            int best = 0;
            float best_value = -1e30f;
            for (int e = 0; e < nn::MoE7FFN::kExperts; ++e) {
                float s = bias[(std::size_t)e];
                for (int i = 0; i < dim; ++i) s += w_r[(std::size_t)i * nn::MoE7FFN::kExperts + e] * x_host[(std::size_t)r * dim + i];
                if (e == 0 || s > best_value) {
                    best_value = s;
                    best = e;
                }
            }
            host_pick[(std::size_t)r] = best;
            host_counts[(std::size_t)best] += 1;
        }
        std::vector<float> y_ref((std::size_t)rows * dim, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < dim; ++i) {
                y_ref[(std::size_t)r * dim + i] =
                    expert_out[(std::size_t)host_pick[(std::size_t)r]][(std::size_t)r * dim + i] +
                    shared_out[(std::size_t)r * dim + i];
            }
        }

        std::vector<float> diff(y_vk.size(), 0.0f);
        for (std::size_t i = 0; i < y_vk.size(); ++i) diff[i] = y_vk[i] - y_ref[i];
        const double denom = std::max(1e-6, max_abs(y_ref));
        const double rel = max_abs(diff) / denom;

        // ---- 2. routing agreement ----
        const bool counts_ok = moe->read_token_counts();
        int count_mismatch = 0;
        if (counts_ok) {
            const auto& load = moe->last_load();
            for (int e = 0; e < nn::MoE7FFN::kExperts; ++e) {
                const int from_module = (int)std::lround(load[(std::size_t)e] * rows);
                if (from_module != host_counts[(std::size_t)e]) ++count_mismatch;
            }
        }

        // ---- 3. gradient semantics ----
        moe->set_training(true);
        auto x2 = Tensor::from_cpu(backend, {rows, dim}, DType::F32, x_host.data());
        auto y2 = moe->forward(x2);
        auto loss = sum_all(y2);
        loss.backward();
        int experts_with_grad = 0;
        int experts_with_grad_not_selected = 0;
        bool router_grad_nonzero = false;
        if (auto g = moe->router_weight.data.grad()) router_grad_nonzero = max_abs(g->to_vector<float>()) > 0.0;
        auto counters_after = moe->counter_layers();
        for (int e = 0; e < nn::MoE7FFN::kExperts; ++e) {
            double vmax = 0.0;
            for (int half = 0; half < 2; ++half) {
                const auto v = counters_after[(std::size_t)(2 * e + half)]->v.to_vector<float>();
                for (float value : v) vmax = std::max(vmax, (double)std::fabs(value));
            }
            if (vmax > 0.0) {
                ++experts_with_grad;
                if (host_counts[(std::size_t)e] == 0) ++experts_with_grad_not_selected;
            }
        }

        std::printf("moe7 reference: rows=%d dim=%d rel_err=%.3e  picks=", rows, dim, rel);
        for (int r = 0; r < rows; ++r) std::printf("%d", host_pick[(std::size_t)r]);
        std::printf("  counts_ok=%d mismatch=%d  router_grad=%d experts_with_grad=%d leaked=%d\n",
                    (int)counts_ok, count_mismatch, (int)router_grad_nonzero, experts_with_grad,
                    experts_with_grad_not_selected);

        const bool pass = rel < 1e-3 && counts_ok && count_mismatch == 0 && router_grad_nonzero &&
                          experts_with_grad >= 1 && experts_with_grad_not_selected == 0;
        if (!pass) {
            std::printf("FAIL test_moe7_reference\n");
            return 1;
        }
        std::printf("PASS test_moe7_reference: forward matches host reference, routing matched, "
                    "gradient reached only routed experts\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL test_moe7_reference: %s\n", e.what());
        return 1;
    }
}
