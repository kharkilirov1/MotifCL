#include <motifcl/nn/compact_counter.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

namespace motifcl::nn {
namespace {

inline std::uint8_t cc_encode(int t, int c, int C) {
    const int lv = 2 * C - 1;
    return static_cast<std::uint8_t>((t + 1) * lv + (c + (C - 1)));
}

// Backward node: recompute the (pre-update) weight on demand for grad_x, then fuse the
// counter update. The decoded weight is a transient -- never stored across the
// forward->backward span -- so peak device memory holds at most one layer's weight at a
// time, not one dense FP32 weight per counter layer. The update runs even when the input
// does not require grad, so the first layer on a raw input still learns.
struct CounterBackwardNode : autograd::Node {
    Tensor x;
    CounterStateLinear* layer;
    std::uint32_t seed;

    CounterBackwardNode(Tensor x_, CounterStateLinear* l, std::uint32_t s)
        : x(std::move(x_)), layer(l), seed(s) {}

    std::vector<Tensor> inputs() const override { return {x}; }

    void backward(const Tensor& grad_out) override {
        const bool need_gx = x.requires_grad();
        // grad_x is computed from the PRE-update state, in-kernel, with no dense weight.
        Tensor grad_x;
        if (need_gx) grad_x = layer->backward_input_from_state(grad_out);
        {
            autograd::NoGradGuard guard;                      // the state update is a side effect
            layer->apply_update_backward(grad_out, x, seed);  // fused: grad_w never materialised
        }
        if (need_gx) x.backward(grad_x);
    }
};

} // namespace

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
    if (backend_->is_vulkan()) {
        const auto result = run_vulkan_compact_counter_decode_weight(
            backend_->vulkan_runtime(),
            state.storage().vulkan_buffer,
            scale.storage().vulkan_buffer,
            w.storage().vulkan_buffer,
            static_cast<std::size_t>(in_),
            static_cast<std::size_t>(out_),
            static_cast<std::size_t>(C_));
        MCL_CHECK(result.success, std::string("vulkan compact-counter decode_weight failed: ") + result.error);
        return w;
    }
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

void CounterStateLinear::apply_update_seed(const Tensor& grad_w, std::uint32_t seed) {
    MCL_CHECK(!backend_->is_vulkan(),
              "CounterStateLinear::apply_update_seed is not Vulkan-native yet; "
              "use apply_update_backward once Vulkan counter update is implemented");
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
    ks.launch1d(static_cast<std::size_t>(out_) * 64, 64);  // one 64-wide work-group per row

    // pass 2 (one work-item per packed group of 4 weights): stochastic-rounding counter tick
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

    // commit the new row scales (apply needed the old ones)
    auto kc = backend_->kernels.get("copy_f32");
    kc.set_arg(0, scale_new.buffer());
    kc.set_arg(1, scale.buffer());
    kc.set_arg(2, out_);
    kc.launch1d(static_cast<std::size_t>(out_));
}

void CounterStateLinear::apply_update_backward(const Tensor& grad_out, const Tensor& x,
                                               std::uint32_t seed) {
    // Memory-native update: grad_w[o,i] = sum_r grad_out[r,o]*x[r,i] is recomputed inside the
    // kernels; no dense [out,in] weight gradient is allocated. Only O(out) scratch (scale_new,
    // denom) is needed -- the device peak holds no weight-sized gradient buffer.
    const int N = x.shape()[0];
    Tensor scale_new = Tensor::empty(*backend_, {out_}, DType::F32);
    Tensor denom = Tensor::empty(*backend_, {out_}, DType::F32);
    if (backend_->is_vulkan()) {
        const auto run = run_vulkan_compact_counter_apply_update_fused(
            backend_->vulkan_runtime(),
            state.storage().vulkan_buffer,
            scale.storage().vulkan_buffer,
            v.storage().vulkan_buffer,
            grad_out.storage().vulkan_buffer,
            x.storage().vulkan_buffer,
            scale_new.storage().vulkan_buffer,
            denom.storage().vulkan_buffer,
            static_cast<std::size_t>(C_),
            static_cast<std::size_t>(in_),
            static_cast<std::size_t>(out_),
            static_cast<std::size_t>(N),
            lr_,
            lr_scale_,
            rms_beta_,
            rms_eps_,
            seed);
        MCL_CHECK(run.success, std::string("vulkan compact-counter fused update failed: ") + run.error);
        return;
    }

    // pass 1: fused row stats (one 64-wide work-group per output row)
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
    ks.launch1d(static_cast<std::size_t>(out_) * 64, 64);

    // pass 2: fused SR counter tick (one wi per packed group of 4 weights)
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

Tensor CounterStateLinear::backward_input_from_state(const Tensor& grad_out) const {
    // grad_x[r,i] = sum_o grad_out[r,o]*scale[o]*t[o,i], decoding t from packed state in the
    // kernel. No dense [out,in] weight is materialised -- only the grad_x [N,in] output.
    const int N = grad_out.shape()[0];
    Tensor grad_x = Tensor::empty(*backend_, {N, in_}, DType::F32);
    if (backend_->is_vulkan()) {
        MCL_CHECK(grad_out.backend_ptr() == backend_, "CounterStateLinear backward_input backend mismatch");
        const auto result = run_vulkan_compact_counter_backward_input_u8(
            backend_->vulkan_runtime(),
            state.storage().vulkan_buffer,
            scale.storage().vulkan_buffer,
            grad_out.storage().vulkan_buffer,
            grad_x.storage().vulkan_buffer,
            static_cast<std::size_t>(N),
            static_cast<std::size_t>(in_),
            static_cast<std::size_t>(out_),
            static_cast<std::size_t>(C_));
        MCL_CHECK(result.success, std::string("vulkan compact-counter backward_input failed: ") + result.error);
        return grad_x;
    }
    auto k = backend_->kernels.get("counter_backward_input_f32");
    k.set_arg(0, grad_out.buffer());
    k.set_arg(1, state.buffer());
    k.set_arg(2, scale.buffer());
    k.set_arg(3, grad_x.buffer());
    k.set_arg(4, C_);
    k.set_arg(5, in_);
    k.set_arg(6, out_);
    k.set_arg(7, N);
    k.launch1d(static_cast<std::size_t>(N) * in_);
    return grad_x;
}

Tensor CounterStateLinear::forward(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "CounterStateLinear expects rank-2 f32 input");
    MCL_CHECK(x.shape()[1] == in_, "CounterStateLinear input feature mismatch");
    // Compute the forward under NoGrad so the ordinary matmul autograd cannot attach a node
    // that retains the dense decoded weight; the counter layer attaches its own node below.
    Tensor y = [&] {
        autograd::NoGradGuard guard;
        Tensor w = decode_weight();             // [out, in] (transient: freed after the forward matmul)
        return matmul_transpose_b(x, w);        // x @ w^T -> [N, out]
    }();
    if (autograd::is_enabled() && training_) {  // gate on training, not x.requires_grad: a counter
        y.set_requires_grad(true);              // layer must update itself even on a raw (grad-less) input
        y._set_grad_fn(std::make_shared<CounterBackwardNode>(x, this, next_seed()));
    }
    return y;
}

} // namespace motifcl::nn
