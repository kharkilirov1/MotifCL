// Mission Phase D: the sparse (zero-gradient-skipping) MoE7 backward against the
// dense one.
//
// A losing expert receives an identically zero gradient. The dense backward still
// runs that expert's full graph on it: a grad_w GEMM over the whole weight
// matrix, a grad_x GEMM, a SwiGLU backward, and the counter update. Every one of
// those results is exactly zero except one: the counter state's row second moment
// decays (v <- rms_beta * v) because the general update folds
// v <- beta*v + (1-beta)*ss with ss == 0.
//
// The sparse backward skips the graph and applies that decay explicitly. This
// test proves the substitution is exact rather than approximate: same module,
// same weights, same input, one run with the dense backward and one with the
// sparse backward, compared *bit for bit* on
//   - the router gradient,
//   - the input gradient,
//   - every expert's packed counter state (the ternary weights themselves),
//   - every expert's row scale and row second moment.
// A relative-error comparison would hide the interesting failures here (a lost
// state tick, a decay applied twice), so the bar is byte equality.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/moe7.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// One layer's persistent state, copied to the host.
struct LayerState {
    std::vector<std::uint8_t> packed;
    std::vector<float> scale;
    std::vector<float> v;
};

std::vector<std::uint8_t> read_bytes(motifcl::Tensor& t) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(t.storage().vulkan_buffer.nbytes()), 0u);
    t.storage().vulkan_buffer.download(out.data(), out.size());
    return out;
}

std::vector<float> read_floats(motifcl::Tensor& t, std::size_t count) {
    std::vector<float> out(count, 0.0f);
    t.storage().vulkan_buffer.download(out.data(), out.size() * sizeof(float));
    return out;
}

struct Snapshot {
    std::vector<std::uint8_t> router_weight;
    std::vector<std::uint8_t> router_bias;
    std::vector<LayerState> layers;
};

Snapshot snapshot(motifcl::nn::MoE7FFN& moe) {
    Snapshot s;
    s.router_weight = read_bytes(moe.router_weight.data);
    s.router_bias = read_bytes(moe.router_bias);
    for (auto* layer : moe.counter_layers()) {
        LayerState ls;
        ls.packed = read_bytes(layer->state);
        ls.scale = read_floats(layer->scale, static_cast<std::size_t>(layer->out_features()));
        ls.v = read_floats(layer->v, static_cast<std::size_t>(layer->out_features()));
        s.layers.push_back(std::move(ls));
    }
    return s;
}

void restore(motifcl::nn::MoE7FFN& moe, const Snapshot& s) {
    moe.router_weight.data.storage().vulkan_buffer.upload(s.router_weight.data(), s.router_weight.size());
    moe.router_bias.storage().vulkan_buffer.upload(s.router_bias.data(), s.router_bias.size());
    auto layers = moe.counter_layers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        auto* layer = layers[i];
        layer->state.storage().vulkan_buffer.upload(s.layers[i].packed.data(), s.layers[i].packed.size());
        layer->scale.storage().vulkan_buffer.upload(s.layers[i].scale.data(),
                                                    s.layers[i].scale.size() * sizeof(float));
        layer->v.storage().vulkan_buffer.upload(s.layers[i].v.data(), s.layers[i].v.size() * sizeof(float));
    }
}

struct Outcome {
    std::vector<std::uint8_t> router_grad;
    std::vector<std::uint8_t> input_grad;
    Snapshot after;
    double loss = 0.0;
};

Outcome run_step(motifcl::Backend& backend, motifcl::nn::MoE7FFN& moe, motifcl::Tensor& x,
                 const std::vector<float>& upstream, std::int64_t rows, std::int64_t dim) {
    auto& rt = backend.vulkan_runtime();
    moe.router_weight.zero_grad();
    x.zero_grad();
    Outcome out;
    // Forward batch, then the routing read-back, then the backward batch: the
    // same shape the trainer uses, and the only point where the losing experts
    // are known before the backward is recorded.
    if (!rt.batch_begin()) {
        std::printf("FAIL batch_begin (forward)\n");
        ok = false;
        return out;
    }
    auto y = moe.forward(x);
    if (!rt.batch_end().success) {
        std::printf("FAIL batch_end (forward)\n");
        ok = false;
        return out;
    }
    if (!moe.freeze_routing()) {
        std::printf("FAIL freeze_routing\n");
        ok = false;
        return out;
    }
    if (!rt.batch_begin()) {
        std::printf("FAIL batch_begin (backward)\n");
        ok = false;
        return out;
    }
    auto grad = motifcl::Tensor::from_cpu(backend, {rows, dim}, motifcl::DType::F32, upstream.data());
    y.backward(grad);
    if (!rt.batch_end().success) {
        std::printf("FAIL batch_end (backward)\n");
        ok = false;
        return out;
    }
    auto rg = moe.router_weight.grad();
    if (!rg) {
        std::printf("FAIL router gradient missing\n");
        ok = false;
        return out;
    }
    out.router_grad = read_bytes(*rg);
    auto xg = x.grad();
    if (!xg) {
        std::printf("FAIL input gradient missing\n");
        ok = false;
        return out;
    }
    out.input_grad = read_bytes(*xg);
    out.after = snapshot(moe);
    return out;
}

bool same_bytes(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b, const char* what,
                std::size_t index) {
    if (a.size() != b.size()) {
        std::printf("FAIL %s[%zu]: size %zu vs %zu\n", what, index, a.size(), b.size());
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::printf("FAIL %s[%zu]: first difference at byte %zu (%02x vs %02x)\n", what, index, i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    using namespace motifcl;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("SKIP test_vulkan_moe7_zero_grad: no Vulkan device\n");
        return 77;
    }

    const int dim = 32, h_exp = 16, h_sh = 16;
    const std::int64_t rows = 24;
    nn::MoE7Config cfg;
    cfg.dim = dim;
    cfg.hidden_expert = h_exp;
    cfg.hidden_shared = h_sh;
    cfg.C = 11;
    cfg.counter_lr = 4e-3f;
    cfg.rms_beta = 0.9f;
    cfg.seed = 20261022u;

    auto backend = Backend::create_vulkan();
    // Two freshly constructed modules: identical buffers *and* identical counter
    // RNG streams (a counter layer's seed advances once per forward, so reusing
    // one module for both runs would compare different stochastic-rounding
    // streams instead of the two backward implementations).
    auto dense_moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
    auto sparse_moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
    dense_moe->set_training(true);
    sparse_moe->set_training(true);

    std::vector<float> x_host(static_cast<std::size_t>(rows * dim));
    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto& v : x_host) v = gauss(rng);
    auto x = Tensor::from_cpu(backend, {rows, dim}, DType::F32, x_host.data());
    x.set_requires_grad(true);

    std::vector<float> upstream(static_cast<std::size_t>(rows * dim));
    for (auto& v : upstream) v = gauss(rng);

    const auto initial = snapshot(*dense_moe);
    restore(*sparse_moe, initial);

    dense_moe->set_skip_zero_grad_experts(false);
    const auto dense = run_step(backend, *dense_moe, x, upstream, rows, dim);
    if (!ok) return 1;

    x.zero_grad();
    sparse_moe->set_skip_zero_grad_experts(true);
    const auto sparse = run_step(backend, *sparse_moe, x, upstream, rows, dim);
    if (!ok) return 1;

    expect(same_bytes(dense.router_grad, sparse.router_grad, "router grad", 0),
           "router gradient is bit-identical (dense vs sparse backward)");
    expect(same_bytes(dense.input_grad, sparse.input_grad, "input grad", 0),
           "input gradient is bit-identical (dense vs sparse backward)");

    auto layers = dense_moe->counter_layers();
    bool state_ok = true, scale_ok = true, v_ok = true;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        state_ok = same_bytes(dense.after.layers[i].packed, sparse.after.layers[i].packed, "state", i) && state_ok;
        std::vector<std::uint8_t> ds(dense.after.layers[i].scale.size() * sizeof(float));
        std::vector<std::uint8_t> ss(sparse.after.layers[i].scale.size() * sizeof(float));
        std::memcpy(ds.data(), dense.after.layers[i].scale.data(), ds.size());
        std::memcpy(ss.data(), sparse.after.layers[i].scale.data(), ss.size());
        scale_ok = same_bytes(ds, ss, "scale", i) && scale_ok;
        std::vector<std::uint8_t> dv(dense.after.layers[i].v.size() * sizeof(float));
        std::vector<std::uint8_t> sv(sparse.after.layers[i].v.size() * sizeof(float));
        std::memcpy(dv.data(), dense.after.layers[i].v.data(), dv.size());
        std::memcpy(sv.data(), sparse.after.layers[i].v.data(), sv.size());
        v_ok = same_bytes(dv, sv, "v", i) && v_ok;
    }
    expect(state_ok, "packed counter state is bit-identical for all 16 layers");
    expect(scale_ok, "row scale is bit-identical for all 16 layers");
    expect(v_ok, "row second moment (v) is bit-identical for all 16 layers");

    // One step is not enough to exercise the interesting case: v starts at zero,
    // so a losing expert's decay is a no-op until that expert has won at least
    // once. Run several more steps on both modules in lockstep and keep comparing.
    // An expert that wins on one step and loses on the next is exactly where a
    // skipped decay would show up.
    bool state_ok_multi = true, v_ok_multi = true, decay_seen = false;
    auto prev_v = dense.after;
    for (int step = 0; step < 4; ++step) {
        // A fresh input per step: with a fixed input the routing never changes, so
        // an expert that wins once keeps winning and the decay never fires.
        for (auto& value : x_host) value = gauss(rng);
        auto step_x = Tensor::from_cpu(backend, {rows, dim}, DType::F32, x_host.data());
        step_x.set_requires_grad(true);
        const auto d = run_step(backend, *dense_moe, step_x, upstream, rows, dim);
        step_x.zero_grad();
        const auto s = run_step(backend, *sparse_moe, step_x, upstream, rows, dim);
        if (!ok) return 1;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            state_ok_multi = same_bytes(d.after.layers[i].packed, s.after.layers[i].packed, "state(step)", i) &&
                             state_ok_multi;
            for (std::size_t r = 0; r < d.after.layers[i].v.size(); ++r) {
                const float before = prev_v.layers[i].v[r];
                const float after = d.after.layers[i].v[r];
                if (before > 0.0f && after < before) decay_seen = true;
                if (after != s.after.layers[i].v[r]) v_ok_multi = false;
            }
        }
        prev_v = d.after;
        expect(same_bytes(d.router_grad, s.router_grad, "router grad(step)", 0) &&
                   same_bytes(d.input_grad, s.input_grad, "input grad(step)", 0),
               "step N: gradients stay bit-identical");
    }
    {
        float v_max = 0.0f;
        std::size_t nonzero = 0;
        for (const auto& layer : dense.after.layers) {
            for (float value : layer.v) {
                v_max = std::max(v_max, std::fabs(value));
                if (value != 0.0f) ++nonzero;
            }
        }
        std::printf("  diagnostic: max |v| = %.6e over %zu non-zero entries\n", (double)v_max, nonzero);
    }
    expect(state_ok_multi, "4 further steps: packed state stays bit-identical");
    expect(v_ok_multi, "4 further steps: v stays bit-identical");
    expect(decay_seen, "an expert that won and then lost did have its v decayed (the case the skip must preserve)");

    std::printf("\n%s\n", ok ? "all zero-gradient backward checks passed" : "FAILURES");
    return ok ? 0 : 1;
}
