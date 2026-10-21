// Mission Phase B: correctness witnesses for the sparse-MoE semantics.
//
// Part 1 (host-only, no GPU): the CPU reference in moe_sparse_reference.hpp is
// exercised on the routing edge cases the sparse path must survive — ties,
// empty experts, all-to-one, uniform, negative/large logits, top-k weight
// normalization, and assignment-list capacity. These run everywhere.
//
// Part 2 (Vulkan, SKIP when unavailable): the shipped dense nn::MoE7FFN
// forward is locked against the same CPU reference on the device, with the
// routing distribution cross-checked. This is the equivalence gate the sparse
// path must also pass (see test_vulkan_moe_sparse_path).
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/moe7.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "moe_sparse_reference.hpp"

namespace {
bool ok = true;
void expect(bool cond, const char* msg) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", msg);
    std::fflush(stdout);
    if (!cond) ok = false;
}
double max_abs(const std::vector<float>& v) {
    double m = 0.0;
    for (float x : v) m = std::max(m, static_cast<double>(std::fabs(x)));
    return m;
}

// ---------------------------------------------------------------- Part 1 ----
void host_reference_tests() {
    using namespace motifcl_test;
    const int dim = 4, h_exp = 3, h_sh = 3, rows = 12;
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(static_cast<std::size_t>(rows) * dim);
    for (auto& v : x) v = gauss(rng);
    std::vector<float> router_w(static_cast<std::size_t>(dim) * 7);
    for (auto& v : router_w) v = gauss(rng);
    std::vector<std::vector<float>> up_w(7), down_w(7);
    for (int e = 0; e < 7; ++e) {
        up_w[e].resize(static_cast<std::size_t>(2 * h_exp) * dim);
        down_w[e].resize(static_cast<std::size_t>(dim) * h_exp);
        for (auto& v : up_w[e]) v = gauss(rng) * 0.5f;
        for (auto& v : down_w[e]) v = gauss(rng) * 0.5f;
    }
    std::vector<float> sh_up(static_cast<std::size_t>(2 * h_sh) * dim), sh_down(static_cast<std::size_t>(dim) * h_sh);
    for (auto& v : sh_up) v = gauss(rng) * 0.5f;
    for (auto& v : sh_down) v = gauss(rng) * 0.5f;

    // 1a. Tie-breaking: identical router rows must all pick expert 0 (lowest index).
    {
        std::vector<float> flat(static_cast<std::size_t>(dim) * 7, 0.25f);  // all logits equal
        std::vector<float> bias(7, 0.0f);
        auto res = moe7_reference_forward(x, rows, dim, flat, bias, up_w, down_w, sh_up, sh_down, h_exp, h_sh);
        bool all_zero = true;
        for (int r = 0; r < rows; ++r) all_zero = all_zero && res.picks[r] == 0;
        expect(all_zero, "tie-break: equal logits select the lowest expert index");
        expect(res.counts[0] == rows, "tie-break: all tokens counted on expert 0");
    }
    // 1b. Bias shifts selection only: a large bias on a never-winning expert moves every token.
    {
        std::vector<float> flat(static_cast<std::size_t>(dim) * 7, 0.0f);
        std::vector<float> bias{0, 0, 0, 0, 0, 0, 100.0f};
        auto res = moe7_reference_forward(x, rows, dim, flat, bias, up_w, down_w, sh_up, sh_down, h_exp, h_sh);
        bool all_six = true;
        for (int r = 0; r < rows; ++r) all_six = all_six && res.picks[r] == 6;
        expect(all_six, "bias: selection follows the bias term");
    }
    // 1c. All-to-one and empty experts: one dominant expert, six empty.
    // Uses a strictly positive input so the dominant logit stays above the
    // zero logits of the other experts for every row.
    {
        std::vector<float> x_pos(static_cast<std::size_t>(rows) * dim);
        for (std::size_t i = 0; i < x_pos.size(); ++i) x_pos[i] = 0.5f + std::fabs(gauss(rng));
        std::vector<float> router(static_cast<std::size_t>(dim) * 7, 0.0f);
        for (int i = 0; i < dim; ++i) router[static_cast<std::size_t>(i) * 7 + 3] = 1.0f;  // expert 3 always wins
        std::vector<float> bias(7, 0.0f);
        auto res = moe7_reference_forward(x_pos, rows, dim, router, bias, up_w, down_w, sh_up, sh_down, h_exp, h_sh);
        expect(res.counts[3] == rows, "all-to-one: dominant expert receives every token");
        bool others_empty = true;
        for (int e = 0; e < 7; ++e)
            if (e != 3) others_empty = others_empty && res.counts[e] == 0;
        expect(others_empty, "all-to-one: the other six experts receive zero tokens");
    }
    // 1d. Uniform-ish split: counts sum to rows, y finite.
    {
        auto res = moe7_reference_forward(x, rows, dim, router_w, std::vector<float>(7, 0.0f), up_w, down_w,
                                          sh_up, sh_down, h_exp, h_sh);
        std::int64_t total = 0;
        for (int e = 0; e < 7; ++e) total += res.counts[e];
        expect(total == rows, "counts: per-expert token counts sum to rows");
        bool finite = true;
        for (float v : res.y) finite = finite && std::isfinite(v);
        expect(finite, "output: finite for random weights");
    }
    // 1e. Negative and large logits: selection must not depend on logit sign/scale.
    // Positive input again, so +80 * sum(x) > 0 > -80 * sum(x) for every row.
    {
        std::vector<float> x_pos(static_cast<std::size_t>(rows) * dim);
        for (std::size_t i = 0; i < x_pos.size(); ++i) x_pos[i] = 0.5f + std::fabs(gauss(rng));
        std::vector<float> router(static_cast<std::size_t>(dim) * 7);
        for (int i = 0; i < dim; ++i)
            for (int e = 0; e < 7; ++e)
                router[static_cast<std::size_t>(i) * 7 + e] = (e == 5 ? -80.0f : (e == 2 ? 80.0f : 0.0f));
        std::vector<float> bias{0, 0, 0, 0, 0, 0, 0};
        auto res = moe7_reference_forward(x_pos, rows, dim, router, bias, up_w, down_w, sh_up, sh_down, h_exp, h_sh);
        bool all_two = true;
        for (int r = 0; r < rows; ++r) all_two = all_two && res.picks[r] == 2;
        expect(all_two, "extreme logits: +80 logit wins over 0 and -80");
        expect(res.counts[2] == rows && res.counts[5] == 0, "extreme logits: counts follow selection");
    }
    // 1f. Single token (T=1) and repeated calls are deterministic.
    {
        std::vector<float> one(x.begin(), x.begin() + dim);
        auto a = moe7_reference_forward(one, 1, dim, router_w, std::vector<float>(7, 0.0f), up_w, down_w,
                                        sh_up, sh_down, h_exp, h_sh);
        auto b = moe7_reference_forward(one, 1, dim, router_w, std::vector<float>(7, 0.0f), up_w, down_w,
                                        sh_up, sh_down, h_exp, h_sh);
        expect(a.picks[0] == b.picks[0] && a.y == b.y, "T=1: repeated calls identical");
    }

    // 1g. General top-k reference: weights normalize, indices distinct, order by logit.
    {
        const int in = 5, hidden = 4, experts = 8, top_k = 2, r2 = 6;
        std::vector<float> x2(static_cast<std::size_t>(r2) * in);
        for (auto& v : x2) v = gauss(rng);
        std::vector<float> rw(static_cast<std::size_t>(in) * experts);
        for (auto& v : rw) v = gauss(rng);
        std::vector<float> gw(static_cast<std::size_t>(experts) * in * hidden), uw(gw.size()),
            dw(static_cast<std::size_t>(experts) * hidden * in);
        for (auto& v : gw) v = gauss(rng) * 0.3f;
        for (auto& v : uw) v = gauss(rng) * 0.3f;
        for (auto& v : dw) v = gauss(rng) * 0.3f;
        auto res = moe_topk_reference_forward(x2, r2, in, rw, gw, uw, dw, experts, hidden, top_k);
        bool weights_ok = true, distinct = true, ordered = true, finite = true;
        for (int r = 0; r < r2; ++r) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) sum += res.weights[static_cast<std::size_t>(r) * top_k + k];
            weights_ok = weights_ok && std::fabs(sum - 1.0f) < 1e-5f;
            for (int k = 0; k < top_k; ++k)
                for (int j = k + 1; j < top_k; ++j)
                    distinct = distinct && res.indices[static_cast<std::size_t>(r) * top_k + k] !=
                                             res.indices[static_cast<std::size_t>(r) * top_k + j];
            // The reference selects in descending-logit order; verify by recomputing logits.
            std::vector<float> logits(experts, 0.0f);
            for (int e = 0; e < experts; ++e) {
                float s = 0.0f;
                for (int i = 0; i < in; ++i)
                    s += x2[static_cast<std::size_t>(r) * in + i] * rw[static_cast<std::size_t>(i) * experts + e];
                logits[e] = s;
            }
            for (int k = 0; k + 1 < top_k; ++k)
                ordered = ordered && logits[res.indices[static_cast<std::size_t>(r) * top_k + k]] >=
                                        logits[res.indices[static_cast<std::size_t>(r) * top_k + k + 1]];
        }
        for (float v : res.y) finite = finite && std::isfinite(v);
        expect(weights_ok, "top-k: softmax over selected logits sums to 1");
        expect(distinct, "top-k: no expert selected twice per token");
        expect(ordered, "top-k: selection ordered by descending logit");
        expect(finite, "top-k: finite output");

        // top_k=1 must equal the single best expert with weight exactly 1.
        auto res1 = moe_topk_reference_forward(x2, r2, in, rw, gw, uw, dw, experts, hidden, 1);
        bool one_ok = true;
        for (int r = 0; r < r2; ++r) one_ok = one_ok && std::fabs(res1.weights[r] - 1.0f) < 1e-6f;
        expect(one_ok, "top-k=1: router weight is exactly 1");
        // max supported top_k = experts degenerates to a full softmax-weighted sum.
        auto resmax = moe_topk_reference_forward(x2, r2, in, rw, gw, uw, dw, experts, hidden, experts);
        bool max_ok = true;
        for (std::size_t i = 0; i < resmax.y.size(); ++i) max_ok = max_ok && std::isfinite(resmax.y[i]);
        expect(max_ok, "top-k=experts: finite full-softmax output");
    }
    std::printf("host reference tests done\n");
}

// ---------------------------------------------------------------- Part 2 ----
int gpu_reference_lock() {
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("SKIP test_vulkan_moe_sparse: no Vulkan device\n");
        return 77;
    }
    Backend backend = Backend::create_vulkan();

    const int dim = 32, h_exp = 16, h_sh = 16, rows = 24;
    nn::MoE7Config cfg;
    cfg.dim = dim;
    cfg.hidden_expert = h_exp;
    cfg.hidden_shared = h_sh;
    cfg.C = 11;
    cfg.counter_lr = 4e-3f;
    cfg.seed = 20261021u;
    auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);

    std::vector<float> x_host(static_cast<std::size_t>(rows) * dim);
    std::mt19937 rng(21);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto& v : x_host) v = gauss(rng);

    moe->set_training(false);
    auto x = Tensor::from_cpu(backend, {rows, dim}, DType::F32, x_host.data());
    auto y = moe->forward(x);
    const auto y_vk = y.to_vector<float>();

    auto counters = moe->counter_layers();
    std::vector<std::vector<float>> up_w(nn::MoE7FFN::kExperts), down_w(nn::MoE7FFN::kExperts);
    for (int e = 0; e < nn::MoE7FFN::kExperts; ++e) {
        up_w[e] = counters[static_cast<std::size_t>(2 * e)]->decode_weight().to_vector<float>();
        down_w[e] = counters[static_cast<std::size_t>(2 * e + 1)]->decode_weight().to_vector<float>();
    }
    const auto sh_up = counters[static_cast<std::size_t>(2 * nn::MoE7FFN::kExperts)]->decode_weight().to_vector<float>();
    const auto sh_dn =
        counters[static_cast<std::size_t>(2 * nn::MoE7FFN::kExperts + 1)]->decode_weight().to_vector<float>();
    const auto w_r = moe->router_weight.data.to_vector<float>();
    const auto bias = moe->router_bias.to_vector<float>();

    const auto ref = motifcl_test::moe7_reference_forward(x_host, rows, dim, w_r, bias, up_w, down_w, sh_up,
                                                          sh_dn, h_exp, h_sh);
    std::vector<float> diff(y_vk.size(), 0.0f);
    for (std::size_t i = 0; i < y_vk.size(); ++i) diff[i] = y_vk[i] - ref.y[i];
    // atol/rtol: the device GEMM accumulates over K in a different order than the
    // host triple loop; 1e-3 relative to max|y| matches the established
    // test_moe7_reference bar (rel_err 1.17e-07 measured there on the same op).
    const double denom = std::max(1e-6, max_abs(ref.y));
    const double rel = max_abs(diff) / denom;
    std::printf("moe dense vs reference: rel_err=%.3e\n", rel);
    expect(rel < 1e-3, "dense MoE7FFN forward matches the CPU reference (rel < 1e-3)");

    // Routing distribution cross-check through the module's own read-back.
    const bool counts_ok = moe->read_token_counts();
    int mismatch = 0;
    if (counts_ok) {
        const auto& load = moe->last_load();
        for (int e = 0; e < nn::MoE7FFN::kExperts; ++e) {
            const int from_module = static_cast<int>(std::lround(load[static_cast<std::size_t>(e)] * rows));
            if (from_module != static_cast<int>(ref.counts[static_cast<std::size_t>(e)])) ++mismatch;
        }
    }
    expect(counts_ok && mismatch == 0, "module token counts match the reference argmax distribution");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    host_reference_tests();
    const int rc = gpu_reference_lock();
    if (rc == 77) return 0;  // skipped: host-only part already ran
    if (!ok || rc != 0) {
        std::printf("FAIL test_vulkan_moe_sparse\n");
        return 1;
    }
    std::printf("PASS test_vulkan_moe_sparse: reference semantics + dense device parity\n");
    return 0;
}
