#include <motifcl/nn/compact_counter.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/runtime/backend.hpp>

namespace motifcl::nn {
namespace {

inline std::uint8_t cc_encode(int t, int c, int C) {
    const int lv = 2 * C - 1;
    return static_cast<std::uint8_t>((t + 1) * lv + (c + (C - 1)));
}

// --- Memory truth gate instrumentation ---------------------------------------
// g_live_dense_w  : dense [out,in] weights currently alive (incremented by the
//                   RAII guard around each transient decode, decremented on scope
//                   exit). g_peak_dense_w tracks the high-water mark.
// g_gradw_allocs  : number of dense [out,in] grad_w buffers that have been fed to
//                   the counter update. The fused path never allocates one.
std::atomic<int> g_live_dense_w{0};
std::atomic<int> g_peak_dense_w{0};
std::atomic<int> g_gradw_allocs{0};

// Scope guard: a dense [out,in] weight is alive for the lifetime of this object.
// The new backward node does NOT retain the weight, so across a forward/backward
// sweep at most one is alive at a time (peak == 1). The old design retained one
// weight per layer in the node, which would push the peak to the network depth.
struct DenseWeightLiveGuard {
    DenseWeightLiveGuard() {
        const int live = g_live_dense_w.fetch_add(1) + 1;
        int prev = g_peak_dense_w.load();
        while (live > prev && !g_peak_dense_w.compare_exchange_weak(prev, live)) {
        }
    }
    ~DenseWeightLiveGuard() { g_live_dense_w.fetch_sub(1); }
    DenseWeightLiveGuard(const DenseWeightLiveGuard&) = delete;
    DenseWeightLiveGuard& operator=(const DenseWeightLiveGuard&) = delete;
};

// Backward node: form grad_x with the OLD weight (decoded as a short-lived
// temporary), then fuse the counter update straight from (grad_out, x) without
// ever materializing a dense weight-gradient. The node stores only x -- no
// pre-decoded weight is carried across forward->backward.
struct CounterBackwardNode : autograd::Node {
    Tensor x;
    CounterStateLinear* layer;
    std::uint32_t seed;

    CounterBackwardNode(Tensor x_, CounterStateLinear* l, std::uint32_t s)
        : x(std::move(x_)), layer(l), seed(s) {}

    std::vector<Tensor> inputs() const override { return {x}; }

    void backward(const Tensor& grad_out) override {
        Tensor grad_x;
        {
            // Transient pre-update weight: decoded, used for grad_x, freed here.
            DenseWeightLiveGuard live;
            Tensor w = layer->decode_weight();          // OLD state (pre-update)
            grad_x = matmul(grad_out, w);               // [N,out] @ [out,in] -> [N,in]
        }
        {
            autograd::NoGradGuard guard;                // update is a side effect
            if (layer->update_enabled())
                layer->apply_update_fused(grad_out, x, seed);  // grad_w computed in-kernel
        }
        if (x.requires_grad()) x.backward(grad_x);
    }
};

} // namespace

void CounterStateLinear::reset_memory_counters() {
    g_live_dense_w.store(0);
    g_peak_dense_w.store(0);
    g_gradw_allocs.store(0);
}
int CounterStateLinear::peak_live_dense_weight() { return g_peak_dense_w.load(); }
int CounterStateLinear::dense_gradw_allocations() { return g_gradw_allocs.load(); }

CounterStateLinear::CounterStateLinear(Backend& backend, int in_features, int out_features,
                                       int C, float lr, float lr_scale, float init_gain,
                                       float rms_beta, float rms_eps, std::uint32_t seed)
    : backend_(&backend), in_(in_features), out_(out_features), C_(C), lr_(lr),
      lr_scale_(lr_scale), rms_beta_(rms_beta), rms_eps_(rms_eps), seed_(seed) {
    MCL_CHECK(in_ > 0 && out_ > 0, "CounterStateLinear requires positive dimensions");
    MCL_CHECK(3 * (2 * C_ - 1) <= 64, "C too large for 6-bit packed state encoding");
    MCL_CHECK(in_ % 4 == 0, "CounterStateLinear requires in_features divisible by 4 (6-bit packing)");

    // state is packed: 4 finite-state codes per 3 bytes, laid out [out, in/4, 3].
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tern(-1, 1);
    const int gpr = in_ / 4;
    std::vector<std::uint8_t> st(static_cast<std::size_t>(out_) * gpr * 3);
    for (int row = 0; row < out_; ++row)
        for (int g = 0; g < gpr; ++g) {
            int code[4];
            for (int k = 0; k < 4; ++k) code[k] = cc_encode(tern(rng), 0, C_);
            std::uint8_t* p = st.data() + (static_cast<std::size_t>(row) * gpr + g) * 3;
            p[0] = static_cast<std::uint8_t>((code[0] & 0x3F) | ((code[1] & 0x03) << 6));
            p[1] = static_cast<std::uint8_t>(((code[1] >> 2) & 0x0F) | ((code[2] & 0x0F) << 4));
            p[2] = static_cast<std::uint8_t>(((code[2] >> 4) & 0x03) | ((code[3] & 0x3F) << 2));
        }
    state = Tensor::from_cpu(*backend_, {out_ * gpr * 3}, DType::U8, st.data());

    const float s0 = init_gain * std::sqrt(3.0f / (2.0f * static_cast<float>(in_)));
    std::vector<float> sc(static_cast<std::size_t>(out_), s0);
    std::vector<float> vv(static_cast<std::size_t>(out_), 0.0f);
    scale = Tensor::from_cpu(*backend_, {out_}, DType::F32, sc.data());
    v = Tensor::from_cpu(*backend_, {out_}, DType::F32, vv.data());
}

Tensor CounterStateLinear::decode_weight() const {
    Tensor w = Tensor::empty(*backend_, {out_, in_}, DType::F32);
    const int n_groups = out_ * (in_ / 4);
    auto k = backend_->kernels.get("decode_counter_state_weight_f32");
    k.set_arg(0, state.buffer());
    k.set_arg(1, scale.buffer());
    k.set_arg(2, w.buffer());
    k.set_arg(3, C_);
    k.set_arg(4, in_);
    k.set_arg(5, n_groups);
    k.launch1d(static_cast<std::size_t>(n_groups));
    return w;
}

void CounterStateLinear::apply_update_fused(const Tensor& grad_out, const Tensor& x,
                                            std::uint32_t seed) {
    // grad_out is [N, out], x is [N, in]. grad_w[o,i] = sum_n grad_out[n,o]*x[n,i] is
    // formed on the fly inside both kernels; no dense [out,in] grad_w is allocated.
    MCL_CHECK(x.ndim() == 2 && grad_out.ndim() == 2, "fused update expects rank-2 grad_out/x");
    const int N = x.shape()[0];
    MCL_CHECK(grad_out.shape()[0] == N && grad_out.shape()[1] == out_ && x.shape()[1] == in_,
              "fused update shape mismatch");

    // pass 1 (one work-group per output row): row-RMS second moment + new scale + denom
    Tensor scale_new = Tensor::empty(*backend_, {out_}, DType::F32);
    Tensor denom = Tensor::empty(*backend_, {out_}, DType::F32);
    auto ks = backend_->kernels.get("counter_row_stats_fused_f32");
    ks.set_arg(0, state.buffer());
    ks.set_arg(1, scale.buffer());
    ks.set_arg(2, v.buffer());
    ks.set_arg(3, grad_out.buffer());
    ks.set_arg(4, x.buffer());
    ks.set_arg(5, scale_new.buffer());
    ks.set_arg(6, denom.buffer());
    ks.set_arg(7, C_);
    ks.set_arg(8, in_);
    ks.set_arg(9, out_);
    ks.set_arg(10, N);
    ks.set_arg(11, lr_scale_);
    ks.set_arg(12, rms_beta_);
    ks.set_arg(13, rms_eps_);
    ks.launch1d(static_cast<std::size_t>(out_) * 64, 64);  // one 64-wide work-group per row

    // pass 2 (one work-item per packed group of 4 weights): SR counter tick
    const int n_groups = out_ * (in_ / 4);
    auto ka = backend_->kernels.get("counter_apply_update_fused_f32");
    ka.set_arg(0, state.buffer());
    ka.set_arg(1, scale.buffer());
    ka.set_arg(2, scale_new.buffer());
    ka.set_arg(3, denom.buffer());
    ka.set_arg(4, grad_out.buffer());
    ka.set_arg(5, x.buffer());
    ka.set_arg(6, C_);
    ka.set_arg(7, in_);
    ka.set_arg(8, out_);
    ka.set_arg(9, N);
    ka.set_arg(10, n_groups);
    ka.set_arg(11, lr_);
    ka.set_arg(12, seed);
    ka.launch1d(static_cast<std::size_t>(n_groups));

    // commit the new row scales (apply needed the old ones)
    auto kc = backend_->kernels.get("copy_f32");
    kc.set_arg(0, scale_new.buffer());
    kc.set_arg(1, scale.buffer());
    kc.set_arg(2, out_);
    kc.launch1d(static_cast<std::size_t>(out_));
}

void CounterStateLinear::apply_update_seed(const Tensor& grad_w, std::uint32_t seed) {
    // Reference path: consumes a dense [out,in] grad_w. Materializing that buffer is
    // the leak the memory truth gate forbids in the training graph, hence the counter
    // bump. Kept only so a test can cross-check the fused path against this one.
    g_gradw_allocs.fetch_add(1);

    // pass 1 (one work-item per output row): row-RMS second moment + new scale + denom
    Tensor scale_new = Tensor::empty(*backend_, {out_}, DType::F32);
    Tensor denom = Tensor::empty(*backend_, {out_}, DType::F32);
    auto ks = backend_->kernels.get("counter_row_stats_f32");
    ks.set_arg(0, state.buffer());
    ks.set_arg(1, scale.buffer());
    ks.set_arg(2, v.buffer());
    ks.set_arg(3, grad_w.buffer());
    ks.set_arg(4, scale_new.buffer());
    ks.set_arg(5, denom.buffer());
    ks.set_arg(6, C_);
    ks.set_arg(7, in_);
    ks.set_arg(8, out_);
    ks.set_arg(9, lr_scale_);
    ks.set_arg(10, rms_beta_);
    ks.set_arg(11, rms_eps_);
    ks.launch1d(static_cast<std::size_t>(out_) * 64, 64);

    const int n_groups = out_ * (in_ / 4);
    auto ka = backend_->kernels.get("counter_apply_update_f32");
    ka.set_arg(0, state.buffer());
    ka.set_arg(1, scale.buffer());
    ka.set_arg(2, scale_new.buffer());
    ka.set_arg(3, denom.buffer());
    ka.set_arg(4, grad_w.buffer());
    ka.set_arg(5, C_);
    ka.set_arg(6, in_);
    ka.set_arg(7, n_groups);
    ka.set_arg(8, lr_);
    ka.set_arg(9, seed);
    ka.launch1d(static_cast<std::size_t>(n_groups));

    auto kc = backend_->kernels.get("copy_f32");
    kc.set_arg(0, scale_new.buffer());
    kc.set_arg(1, scale.buffer());
    kc.set_arg(2, out_);
    kc.launch1d(static_cast<std::size_t>(out_));
}

Tensor CounterStateLinear::forward(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "CounterStateLinear expects rank-2 f32 input");
    MCL_CHECK(x.shape()[1] == in_, "CounterStateLinear input feature mismatch");

    Tensor y;
    {
        // The decoded weight lives only long enough to produce y; it is not carried
        // into the backward node.
        DenseWeightLiveGuard live;
        Tensor w = decode_weight();                 // [out, in]
        y = matmul_transpose_b(x, w);               // x @ w^T -> [N, out]
    }

    // Attach the self-update node whenever training is active, independent of whether
    // the input itself needs a gradient: a counter layer fed raw input (requires_grad
    // == false) must still update its own state. grad_x is only propagated if x asks.
    if (autograd::is_enabled() && update_enabled_) {
        y.set_requires_grad(true);
        y._set_grad_fn(std::make_shared<CounterBackwardNode>(x, this, next_seed()));
    }
    return y;
}

} // namespace motifcl::nn
