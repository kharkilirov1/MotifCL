#pragma once

#include <cstdint>

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

// Ternary finite-state "counter" linear layer trained as a per-synapse automaton.
// Persistent per-weight state is one uint8 (45/63 reachable states); the optimizer
// lives inside that state plus per-row scale and per-row RMS second moment. The
// weight update is fused into backward (no persistent weight-gradient or Adam moments).
// Layout mirrors the validated prototype: state/grad_w are [out, in]; forward is
// y = x @ w^T via matmul_transpose_b. Eager-training only (in-backward update is a
// side effect; not graph-capture replayable).
class CounterStateLinear : public Module {
public:
    CounterStateLinear(Backend& backend, int in_features, int out_features,
                       int C = 11, float lr = 3e-3f, float lr_scale = 0.0f,
                       float init_gain = 1.0f, float rms_beta = 0.9f,
                       float rms_eps = 1e-3f, std::uint32_t seed = 12345u);

    Tensor forward(const Tensor& x) override;
    Tensor forward_swiglu(const Tensor& x);
    std::vector<Parameter*> parameters() override { return {}; }  // self-updating

    Tensor decode_weight() const;                       // dense [out, in] f32
    void apply_update_seed(const Tensor& grad_w, std::uint32_t seed);  // reference path (dense grad_w)
    // Uses one reusable tiled grad_w workspace for eligible Vulkan layers;
    // the bounded-memory fallback recomputes grad_w inside the update kernels.
    void apply_update_backward(const Tensor& grad_out, const Tensor& x, std::uint32_t seed);
    // State-only update for an identically zero gradient (a MoE expert that lost
    // the router's argmax). With grad_w == 0 the tick and the row scale are
    // provably unchanged and only the row second moment survives:
    // v <- rms_beta * v. Applying it keeps a sparse backward bit-identical to the
    // dense one while skipping the grad_w GEMM and the per-group tick.
    void apply_zero_grad_update();
    // Memory-native grad wrt input: grad_x decoded from packed state in-kernel,
    // without materialising a dense [out,in] weight. Returns grad_x [N,in].
    Tensor backward_input_from_state(const Tensor& grad_out) const;
    std::uint32_t next_seed() { return seed_++; }

    void set_training(bool t) { training_ = t; }   // when false, forward attaches no update node
    bool training() const { return training_; }
    void set_lr(float lr) { lr_ = lr; }            // schedule hook (pretrain runner)
    float lr() const { return lr_; }

    int in_features() const { return in_; }
    int out_features() const { return out_; }

    // Persistent state (saved/restored as buffers, not optimizer-touched).
    Tensor state;   // U8  [out, in]
    Tensor scale;   // F32 [out]
    Tensor v;       // F32 [out]

private:
    // Lazily allocated dense [out,in] FP32 mirror of the packed state. Only the
    // MOTIFCL_COUNTER_DECODE_DENSE A/B path uses it; it is refreshed from the
    // packed state on every call (the state moves every optimizer step) and the
    // buffer is reused so the path does not pay an allocation per projection.
    const Tensor& ensure_dense_weight() const;
    mutable Tensor dense_weight_;

    Backend* backend_ = nullptr;
    int in_ = 0;
    int out_ = 0;
    int C_ = 11;
    float lr_ = 3e-3f;
    float lr_scale_ = 0.0f;
    float rms_beta_ = 0.9f;
    float rms_eps_ = 1e-3f;
    std::uint32_t seed_ = 12345u;
    bool training_ = true;
};

} // namespace motifcl::nn
