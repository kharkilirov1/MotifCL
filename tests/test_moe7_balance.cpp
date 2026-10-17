// Witness for the auxiliary load-balancing gradient of nn::MoE7FFN.
//
// L_aux = E * sum_e f_e * P_e with f frozen at the observed load and P_e the mean
// router probability, so dL_aux/dlogit_{n,i} = (E/N) * p_{n,i} * (f_i - sum_j f_j p_{n,j}).
// The module computes that on the host, uploads it and turns it into a weight
// gradient with matmul_transpose_a(x, dl). This test recomputes the whole chain
// independently on the host from the public router weight/bias and compares it
// against the gradient the module actually accumulated into the router parameter.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/moe7.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace {
constexpr int kE = 7;
static_assert(kE == motifcl::nn::MoE7FFN::kExperts, "expert count drifted from the module");
}

int main() {
    using namespace motifcl;
    try {
        const auto probe = probe_vulkan_runtime();
        if (!probe.available()) {
            std::printf("SKIP test_moe7_balance: no Vulkan device\n");
            return 0;
        }
        Backend backend = Backend::create_vulkan();

        const int dim = 32, h_exp = 16, h_shared = 16, rows = 16;
        nn::MoE7Config cfg;
        cfg.dim = dim;
        cfg.hidden_expert = h_exp;
        cfg.hidden_shared = h_shared;
        cfg.C = 11;
        cfg.seed = 991u;
        auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
        moe->set_training(false);

        std::vector<float> x_host((std::size_t)rows * dim, 0.0f);
        std::mt19937 rng(3);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (auto& v : x_host) v = gauss(rng);

        auto x = Tensor::from_cpu(backend, {rows, dim}, DType::F32, x_host.data());
        auto y = moe->forward(x);
        (void)y;
        if (!moe->read_token_counts()) {
            std::printf("FAIL test_moe7_balance: read_token_counts returned false\n");
            return 1;
        }

        const auto w_r = moe->router_weight.data.to_vector<float>();
        const auto bias = moe->router_bias.to_vector<float>();
        const auto& load = moe->last_load();

        // Independent host chain: logits -> p -> dl -> gW.
        std::vector<float> dl((std::size_t)rows * kE, 0.0f);
        const float e_over_n = (float)kE / (float)rows;
        for (int r = 0; r < rows; ++r) {
            float logits[kE];
            for (int e = 0; e < kE; ++e) {
                float s = bias[(std::size_t)e];
                for (int i = 0; i < dim; ++i) s += w_r[(std::size_t)i * kE + e] * x_host[(std::size_t)r * dim + i];
                logits[e] = s;
            }
            float p[kE];
            for (int e = 0; e < kE; ++e) p[e] = std::exp(logits[e]);
            float z = 0.0f;
            for (int e = 0; e < kE; ++e) z += p[e];
            for (int e = 0; e < kE; ++e) p[e] /= z;
            float load_dot = 0.0f;
            for (int e = 0; e < kE; ++e) load_dot += load[(std::size_t)e] * p[e];
            for (int i = 0; i < kE; ++i) {
                dl[(std::size_t)r * kE + i] = e_over_n * p[i] * (load[(std::size_t)i] - load_dot);
            }
        }
        const float aux_weight = 1.0f;
        std::vector<float> g_w((std::size_t)dim * kE, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < dim; ++i) {
                const float xv = x_host[(std::size_t)r * dim + i];
                for (int e = 0; e < kE; ++e) {
                    g_w[(std::size_t)i * kE + e] += xv * dl[(std::size_t)r * kE + e] * aux_weight;
                }
            }
        }

        // Module path: host-side dl -> device matmul_transpose_a -> router grad.
        moe->router_weight.data.zero_grad();
        moe->prepare_balance_gradient(aux_weight);
        moe->inject_balance_gradient();
        auto grad = moe->router_weight.data.grad();
        if (!grad) {
            std::printf("FAIL test_moe7_balance: router has no gradient after injection\n");
            return 1;
        }
        const auto g_device = grad->to_vector<float>();
        if (g_device.size() != g_w.size()) {
            std::printf("FAIL test_moe7_balance: gradient size mismatch\n");
            return 1;
        }
        double max_diff = 0.0, max_ref = 0.0;
        for (std::size_t i = 0; i < g_w.size(); ++i) {
            max_diff = std::max(max_diff, (double)std::fabs(g_device[i] - g_w[i]));
            max_ref = std::max(max_ref, (double)std::fabs(g_w[i]));
        }
        const double rel = max_diff / std::max(1e-12, max_ref);

        // Linearity in aux_weight: doubling the weight must double the injected gradient.
        moe->router_weight.data.zero_grad();
        moe->prepare_balance_gradient(aux_weight * 2.0f);
        moe->inject_balance_gradient();
        const auto g2 = moe->router_weight.data.grad()->to_vector<float>();
        double max_scale_err = 0.0;
        for (std::size_t i = 0; i < g_w.size(); ++i) {
            max_scale_err = std::max(max_scale_err, (double)std::fabs(g2[i] - 2.0f * g_w[i]));
        }
        const double rel_scale = max_scale_err / std::max(1e-12, 2.0 * max_ref);

        std::printf("moe7 balance: rel_err=%.3e  rel_err(2x)=%.3e  grad_max=%.3e  loads=",
                    rel, rel_scale, max_ref);
        for (int e = 0; e < kE; ++e) std::printf("%.3f%s", load[(std::size_t)e], e + 1 == kE ? "\n" : " ");

        const bool pass = rel < 1e-3 && rel_scale < 1e-3 && max_ref > 0.0;
        if (!pass) {
            std::printf("FAIL test_moe7_balance\n");
            return 1;
        }
        std::printf("PASS test_moe7_balance: injected router gradient matches the analytic "
                    "Switch auxiliary gradient and scales linearly with AUX_WEIGHT\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL test_moe7_balance: %s\n", e.what());
        return 1;
    }
}
