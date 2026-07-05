#pragma once

#include <motifcl/tensor/tensor.hpp>

#include <vector>

namespace motifcl::autograd {

class Node {
public:
    virtual ~Node() = default;
    virtual std::vector<Tensor> inputs() const { return {}; }
    virtual void backward(const Tensor& grad_output) = 0;
};

bool is_enabled();
void set_enabled(bool enabled);

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
private:
    bool previous_ = true;
};

// RAII scope that detaches the current thread's active backward engine, so a
// Node's backward() can run a NESTED top-level backward (e.g. for recompute-
// based backward used by reversible blocks / activation checkpointing). On
// destruction the previously active engine is restored, so any subsequent
// x.backward(grad) calls inside the surrounding backward re-accumulate into
// the original engine's pending map as expected.
//
// Used by nn::ReversibleBlock. Without this scope, calling Tensor::backward()
// inside a Node's backward() only accumulates into the active engine's pending
// map (src/tensor/tensor.cpp:264), which never visits the freshly-built graph.
class IsolatedBackwardScope {
public:
    IsolatedBackwardScope();
    ~IsolatedBackwardScope();
private:
    void* previous_ = nullptr;  // BackwardEngine* (opaque; defined in tensor.cpp)
};

// Internal accessors used by IsolatedBackwardScope; defined in tensor.cpp
// (where the engine thread-local lives) and called only from the scope's
// constructor/destructor. Do not call directly.
void* _save_and_clear_active_backward_engine();
void _restore_active_backward_engine(void* engine);

} // namespace motifcl::autograd
