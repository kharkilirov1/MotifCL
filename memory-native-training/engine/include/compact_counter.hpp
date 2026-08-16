#pragma once

#include <cstdint>

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

// Ternary finite-state "counter" linear layer trained as a per-synapse automaton.
// Persistent per-weight state is one uint8 (45/63 reachable states); the optimizer
// lives inside that state plus per-row scale and per-row RMS second moment. The
// weight update is fused into backward (no full weight-gradient, no Adam moments).
// Layout mirrors the validated prototype: state is packed [out, in/4, 3]; forward is
// y = x @ w^T via matmul_transpose_b. Eager-training only (in-backward update is a
// side effect; not graph-capture replayable).
//
// Memory-native backward (the point of the method): the dense weight is NOT held
// across forward->backward, and the dense weight-gradient is NEVER materialized.
// In backward the weight is decoded as a short-lived temporary only to form grad_x,
// and the counter update reads grad_w element-by-element straight out of an OpenCL
// kernel (grad_w[o,i] = sum_n grad_out[n,o]*x[n,i] computed on the fly). The training
// peak therefore stays at activations + at most one transient [out,in] weight, not
// one dense weight or one dense grad_w per layer. See tests/test_counter_memory_truth.cpp.
class CounterStateLinear : public Module {
public:
    CounterStateLinear(Backend& backend, int in_features, int out_features,
                       int C = 11, float lr = 3e-3f, float lr_scale = 0.0f,
                       float init_gain = 1.0f, float rms_beta = 0.9f,
                       float rms_eps = 1e-3f, std::uint32_t seed = 12345u);

    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {}; }  // self-updating

    Tensor decode_weight() const;                       // dense [out, in] f32 (transient)

    // Memory-native update: consumes (grad_out, x) and computes grad_w on the fly in
    // the kernel. No dense [out,in] grad_w is ever allocated. This is the default path.
    void apply_update_fused(const Tensor& grad_out, const Tensor& x, std::uint32_t seed);

    // Reference update that materializes a dense [out,in] grad_w. Kept only for
    // cross-checking the fused path; NOT used in the training graph. Allocating the
    // dense grad_w is exactly the leak the memory truth gate forbids, so this bumps
    // the gradw counter.
    void apply_update_seed(const Tensor& grad_w, std::uint32_t seed);

    std::uint32_t next_seed() { return seed_++; }

    void set_update_enabled(bool enabled) { update_enabled_ = enabled; }
    bool update_enabled() const { return update_enabled_; }

    int in_features() const { return in_; }
    int out_features() const { return out_; }

    // --- Memory truth gate instrumentation (host-side, O(1)) ---------------------
    // Counters the truth-gate test reads to prove the backward stays compact:
    //  * peak_live_dense_weight(): max number of dense [out,in] weights alive at once
    //    across a backward pass. Must stay <= 1 (no saved-weight-per-layer leak).
    //  * dense_gradw_allocations(): number of full [out,in] grad_w buffers allocated.
    //    Must be 0 on the fused path (no grad_w leak).
    static void reset_memory_counters();
    static int peak_live_dense_weight();
    static int dense_gradw_allocations();

    // Persistent state (saved/restored as buffers, not optimizer-touched).
    Tensor state;   // U8  packed [out, in/4, 3]
    Tensor scale;   // F32 [out]
    Tensor v;       // F32 [out]

private:
    Backend* backend_ = nullptr;
    int in_ = 0;
    int out_ = 0;
    int C_ = 11;
    float lr_ = 3e-3f;
    float lr_scale_ = 0.0f;
    float rms_beta_ = 0.9f;
    float rms_eps_ = 1e-3f;
    std::uint32_t seed_ = 12345u;
    bool update_enabled_ = true;
};

} // namespace motifcl::nn
