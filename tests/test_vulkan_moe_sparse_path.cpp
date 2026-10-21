// Mission Phase C: the sparse Vulkan MoE7 path against the dense one.
//
// Part 1 (Vulkan, SKIP when unavailable) — equivalence. The same nn::MoE7FFN
// instance, the same weights, the same input: the dense forward computes all
// seven experts, the sparse forward computes only the router's winner. The two
// must agree to GEMM accumulation order (they run the same tile kernel, so the
// bar is tight, not a loose "close enough"), and the routing counts must match
// the reference argmax distribution.
//
// Part 2 — edge cases that the compact-list scheme has to survive: every token
// on one expert (six empty lists), a single token, and a token count that is
// not a multiple of the 32-row tile.
//
// Part 3 — timing at the documented 15M training shape (T=2048, dim=288,
// hidden=192) plus decode (T=1), reported as GPU time per call and per-token
// throughput, with the dispatch count from the runtime profiler.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/moe7.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

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

struct Scratch {
    motifcl::VulkanBuffer logits;
    motifcl::VulkanBuffer counts;
    motifcl::VulkanBuffer list;
    motifcl::VulkanBuffer packed;
    motifcl::VulkanBuffer act;
    motifcl::VulkanBuffer expert_out;
};

Scratch make_scratch(motifcl::VulkanRuntime& rt, std::size_t tokens, std::size_t dim, std::size_t hidden) {
    Scratch s;
    s.logits = rt.create_buffer(tokens * 7 * sizeof(float));
    s.counts = rt.create_buffer(8 * sizeof(std::uint32_t));
    s.list = rt.create_buffer(7 * tokens * sizeof(std::uint32_t));
    s.packed = rt.create_buffer(tokens * 2 * hidden * sizeof(float));
    s.act = rt.create_buffer(tokens * hidden * sizeof(float));
    s.expert_out = rt.create_buffer(tokens * dim * sizeof(float));
    return s;
}

motifcl::VulkanMoE7SparseSpec make_spec(motifcl::nn::MoE7FFN& moe, const motifcl::Tensor& x, Scratch& s,
                                        std::size_t tokens, std::size_t dim, std::size_t hidden) {
    motifcl::VulkanMoE7SparseSpec spec;
    spec.tokens = tokens;
    spec.dim = dim;
    spec.hidden = hidden;
    spec.C = 11;
    spec.x = &x.storage().vulkan_buffer;
    spec.router_w = &moe.router_weight.data.storage().vulkan_buffer;
    spec.router_b = &moe.router_bias.storage().vulkan_buffer;
    auto counters = moe.counter_layers();
    for (int e = 0; e < 7; ++e) {
        spec.up_state[e] = &counters[static_cast<std::size_t>(2 * e)]->state.storage().vulkan_buffer;
        spec.up_scale[e] = &counters[static_cast<std::size_t>(2 * e)]->scale.storage().vulkan_buffer;
        spec.down_state[e] = &counters[static_cast<std::size_t>(2 * e + 1)]->state.storage().vulkan_buffer;
        spec.down_scale[e] = &counters[static_cast<std::size_t>(2 * e + 1)]->scale.storage().vulkan_buffer;
    }
    spec.logits = &s.logits;
    spec.counts = &s.counts;
    spec.list = &s.list;
    spec.packed = &s.packed;
    spec.act = &s.act;
    spec.expert_out = &s.expert_out;
    return spec;
}

// The dense path merges the shared expert inside forward(); the sparse path
// deliberately leaves the merge to the caller, so the test reconstructs it the
// same way nn::MoE7FFN::run_expert(shared_, x) does.
motifcl::Tensor shared_expert_out(motifcl::nn::MoE7FFN& moe, const motifcl::Tensor& x) {
    auto counters = moe.counter_layers();
    auto* up = counters[static_cast<std::size_t>(2 * 7)];
    auto* down = counters[static_cast<std::size_t>(2 * 7 + 1)];
    return down->forward(up->forward_swiglu(x));
}

std::vector<std::uint32_t> read_counts(motifcl::VulkanBuffer& counts) {
    std::vector<std::uint32_t> out(7, 0u);
    counts.download(out.data(), out.size() * sizeof(std::uint32_t));
    return out;
}

std::vector<float> read_floats(motifcl::VulkanBuffer& buffer, std::size_t elements) {
    std::vector<float> out(elements, 0.0f);
    buffer.download(out.data(), out.size() * sizeof(float));
    return out;
}

// ------------------------------------------------------------ equivalence ----
void equivalence_case(std::size_t rows, const char* label) {
    using namespace motifcl;
    const int dim = 32, h_exp = 16, h_sh = 16;
    nn::MoE7Config cfg;
    cfg.dim = dim;
    cfg.hidden_expert = h_exp;
    cfg.hidden_shared = h_sh;
    cfg.C = 11;
    cfg.counter_lr = 4e-3f;
    cfg.seed = 20261021u;

    auto backend = Backend::create_vulkan();
    auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
    moe->set_training(false);

    std::vector<float> x_host(rows * dim);
    std::mt19937 rng(4242);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto& v : x_host) v = gauss(rng);
    auto x = Tensor::from_cpu(backend, {static_cast<std::int64_t>(rows), dim}, DType::F32, x_host.data());

    auto& rt = backend.vulkan_runtime();
    auto scratch = make_scratch(rt, rows, dim, h_exp);

    std::vector<float> dense;
    std::vector<float> shared_host;
    {
        autograd::NoGradGuard guard;
        auto y = moe->forward(x);
        dense = y.to_vector<float>();
        auto sh = shared_expert_out(*moe, x);
        shared_host = sh.to_vector<float>();
    }

    auto spec = make_spec(*moe, x, scratch, rows, dim, h_exp);
    const auto result = run_vulkan_moe7_sparse_forward(rt, spec);
    std::printf("%s: sparse dispatch %s%s\n", label, result.success ? "ok" : "FAILED",
                result.success ? "" : (" (" + result.error + ")").c_str());
    expect(result.success, "sparse forward dispatches successfully");
    if (!result.success) return;

    const auto expert_host = read_floats(scratch.expert_out, rows * dim);
    std::vector<float> sparse_total(expert_host.size());
    for (std::size_t i = 0; i < sparse_total.size(); ++i) sparse_total[i] = expert_host[i] + shared_host[i];

    std::vector<float> diff(sparse_total.size());
    for (std::size_t i = 0; i < diff.size(); ++i) diff[i] = sparse_total[i] - dense[i];
    const double denom = std::max(1e-6, max_abs(dense));
    const double rel = max_abs(diff) / denom;
    std::printf("%s: sparse vs dense rel_err=%.3e (rows=%zu)\n", label, rel, rows);
    // Same tile kernel and same accumulation order on both paths, so the only
    // difference is which rows are visited; anything above fp32 round-off means
    // the gather/scatter addressing is wrong.
    expect(rel < 1e-6, "sparse forward equals the dense forward");

    // Routing distribution must match the module's own argmax read-back.
    const auto counts = read_counts(scratch.counts);
    std::uint64_t total = 0;
    for (auto c : counts) total += c;
    expect(total == rows, "routing counts sum to the token count (no dropped tokens)");
    const bool counts_ok = moe->read_token_counts();
    expect(counts_ok, "module token counts readable");
    if (counts_ok) {
        int mismatch = 0;
        const auto& load = moe->last_load();
        for (int e = 0; e < 7; ++e) {
            const auto from_module = static_cast<std::uint32_t>(std::lround(load[static_cast<std::size_t>(e)] * rows));
            if (from_module != counts[static_cast<std::size_t>(e)]) ++mismatch;
        }
        expect(mismatch == 0, "sparse routing counts match the dense argmax distribution");
    }
}

// ------------------------------------------------------------ single expert --
void all_to_one_case() {
    using namespace motifcl;
    const int dim = 32, h_exp = 16, h_sh = 16;
    const std::size_t rows = 40;  // not a multiple of the 32-row tile
    nn::MoE7Config cfg;
    cfg.dim = dim;
    cfg.hidden_expert = h_exp;
    cfg.hidden_shared = h_sh;
    cfg.C = 11;
    cfg.counter_lr = 4e-3f;
    cfg.seed = 7u;

    auto backend = Backend::create_vulkan();
    auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
    moe->set_training(false);
    // Drive every token to expert 4: the bias shifts selection only, and a bias
    // of +1e3 dwarfs any logit the random router produces.
    std::vector<float> bias(7, 0.0f);
    bias[4] = 1.0e3f;
    moe->router_bias = Tensor::from_cpu(backend, {7}, DType::F32, bias.data());

    std::vector<float> x_host(rows * dim);
    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto& v : x_host) v = gauss(rng);
    auto x = Tensor::from_cpu(backend, {static_cast<std::int64_t>(rows), dim}, DType::F32, x_host.data());

    auto& rt = backend.vulkan_runtime();
    auto scratch = make_scratch(rt, rows, dim, h_exp);
    auto spec = make_spec(*moe, x, scratch, rows, dim, h_exp);
    const auto result = run_vulkan_moe7_sparse_forward(rt, spec);
    expect(result.success, "all-to-one: sparse forward dispatches successfully");
    if (!result.success) return;

    const auto counts = read_counts(scratch.counts);
    bool one_full = counts[4] == rows;
    bool rest_empty = true;
    for (int e = 0; e < 7; ++e)
        if (e != 4) rest_empty = rest_empty && counts[static_cast<std::size_t>(e)] == 0;
    expect(one_full, "all-to-one: the dominant expert receives every token");
    expect(rest_empty, "all-to-one: the other six lists are empty");

    std::vector<float> dense;
    std::vector<float> shared_host;
    {
        autograd::NoGradGuard guard;
        dense = moe->forward(x).to_vector<float>();
        shared_host = shared_expert_out(*moe, x).to_vector<float>();
    }
    const auto expert_host = read_floats(scratch.expert_out, rows * dim);
    std::vector<float> diff(rows * dim);
    for (std::size_t i = 0; i < diff.size(); ++i) diff[i] = expert_host[i] + shared_host[i] - dense[i];
    const double rel = max_abs(diff) / std::max(1e-6, max_abs(dense));
    std::printf("all-to-one: rel_err=%.3e\n", rel);
    expect(rel < 1e-6, "all-to-one: sparse output equals dense output");
}

// ------------------------------------------------------------------ timing ---
struct Timing {
    double gpu_us = 0.0;
    double median_us = 0.0;
    std::uint64_t dispatches = 0;
};

Timing time_dense(motifcl::Backend& backend, motifcl::nn::MoE7FFN& moe, motifcl::Tensor& x, int iters) {
    auto& rt = backend.vulkan_runtime();
    rt.set_gpu_timing_enabled(true);
    std::vector<double> samples;
    std::uint64_t dispatches = 0;
    motifcl::autograd::NoGradGuard guard;
    for (int i = 0; i < iters + 3; ++i) {
        rt.reset_kernel_profile();
        rt.batch_begin();
        auto y = moe.forward(x);
        rt.batch_end();
        if (i < 3) continue;
        samples.push_back(rt.last_gpu_time_us());
        std::uint64_t calls = 0;
        for (const auto& entry : rt.kernel_profile()) calls += entry.calls;
        dispatches = calls;
    }
    std::sort(samples.begin(), samples.end());
    Timing t;
    t.median_us = samples[samples.size() / 2];
    t.gpu_us = t.median_us;
    t.dispatches = dispatches;
    return t;
}

Timing time_sparse(motifcl::Backend& backend, motifcl::VulkanMoE7SparseSpec& spec, int iters) {
    auto& rt = backend.vulkan_runtime();
    rt.set_gpu_timing_enabled(true);
    std::vector<double> samples;
    std::uint64_t dispatches = 0;
    motifcl::autograd::NoGradGuard guard;
    for (int i = 0; i < iters + 3; ++i) {
        rt.reset_kernel_profile();
        rt.batch_begin();
        const auto result = run_vulkan_moe7_sparse_forward(rt, spec);
        rt.batch_end();
        if (!result.success) {
            std::printf("sparse timing dispatch failed: %s\n", result.error.c_str());
            return Timing{};
        }
        if (i < 3) continue;
        samples.push_back(rt.last_gpu_time_us());
        std::uint64_t calls = 0;
        for (const auto& entry : rt.kernel_profile()) calls += entry.calls;
        dispatches = calls;
    }
    std::sort(samples.begin(), samples.end());
    if (const char* want = std::getenv("MOTIFCL_VK_MOE_PROFILE"); want && *want == '1') {
        double sum = 0.0;
        std::printf("  samples us:");
        for (double s : samples) std::printf(" %.0f", s);
        std::printf("\n");
        std::printf("  sparse kernel profile (last call):\n");
        for (const auto& entry : rt.kernel_profile()) {
            sum += entry.total_us;
            if (entry.total_us < 1.0) continue;
            std::printf("    %-34s calls %3llu  %8.1f us\n", entry.name.c_str(),
                        static_cast<unsigned long long>(entry.calls), entry.total_us);
        }
        std::printf("    %-34s            %8.1f us  (wall %.1f us, unattributed %.1f us)\n",
                    "sum of kernel time", sum, rt.last_gpu_time_us(),
                    rt.last_gpu_time_us() - sum);
    }
    Timing t;
    t.median_us = samples[samples.size() / 2];
    t.gpu_us = t.median_us;
    t.dispatches = dispatches;
    return t;
}

void timing_case(std::size_t tokens, const char* label) {
    using namespace motifcl;
    const int dim = 288, h_exp = 192, h_sh = 192;
    nn::MoE7Config cfg;
    cfg.dim = dim;
    cfg.hidden_expert = h_exp;
    cfg.hidden_shared = h_sh;
    cfg.C = 11;
    cfg.counter_lr = 4e-3f;
    cfg.seed = 1337u;

    auto backend = Backend::create_vulkan();
    auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
    moe->set_training(false);

    std::vector<float> x_host(tokens * dim);
    std::mt19937 rng(5);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto& v : x_host) v = gauss(rng);
    auto x = Tensor::from_cpu(backend, {static_cast<std::int64_t>(tokens), dim}, DType::F32, x_host.data());

    auto& rt = backend.vulkan_runtime();
    auto scratch = make_scratch(rt, tokens, dim, h_exp);
    auto spec = make_spec(*moe, x, scratch, tokens, dim, h_exp);

    const int iters = tokens <= 8 ? 60 : 20;
    const auto dense = time_dense(backend, *moe, x, iters);
    const auto sparse = time_sparse(backend, spec, iters);

    const double tok_per_s_dense = tokens * 1.0e6 / std::max(1.0, dense.median_us);
    const double tok_per_s_sparse = tokens * 1.0e6 / std::max(1.0, sparse.median_us);
    std::printf("\n[%s] tokens=%zu dim=%d hidden=%d\n", label, tokens, dim, h_exp);
    std::printf("  dense : median %8.1f us  %9.0f tok/s  dispatches %llu\n", dense.median_us,
                tok_per_s_dense, static_cast<unsigned long long>(dense.dispatches));
    std::printf("  sparse: median %8.1f us  %9.0f tok/s  dispatches %llu\n", sparse.median_us,
                tok_per_s_sparse, static_cast<unsigned long long>(sparse.dispatches));
    std::printf("  speedup %.3fx\n", dense.median_us / std::max(1e-9, sparse.median_us));
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep progress visible if a dispatch crashes
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("SKIP test_vulkan_moe_sparse_path: no Vulkan device\n");
        return 77;
    }
    std::printf("device: %s\n\n", probe.devices.empty() ? "?" : probe.devices[0].name.c_str());

    equivalence_case(24, "T=24");
    equivalence_case(1, "T=1");
    equivalence_case(37, "T=37 (partial tile)");
    all_to_one_case();
    timing_case(2048, "prefill");
    timing_case(1, "decode");

    std::printf("\n%s\n", ok ? "all sparse-path checks passed" : "SPARSE PATH FAILURES");
    return ok ? 0 : 1;
}
