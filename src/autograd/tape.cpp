#include <motifcl/autograd/tape.hpp>

namespace motifcl::autograd {

namespace {
thread_local bool g_autograd_enabled = true;
}

bool is_enabled() { return g_autograd_enabled; }
void set_enabled(bool enabled) { g_autograd_enabled = enabled; }

NoGradGuard::NoGradGuard() : previous_(g_autograd_enabled) { g_autograd_enabled = false; }
NoGradGuard::~NoGradGuard() { g_autograd_enabled = previous_; }

// IsolatedBackwardScope: implemented through internal accessors defined in
// tensor.cpp (where the active-backward-engine thread-local lives). Saving
// captures the engine pointer and clears it; restore puts it back. The
// engine pointer itself is opaque (void*) here so this TU does not need to
// see BackwardEngine internals.
IsolatedBackwardScope::IsolatedBackwardScope()
    : previous_(_save_and_clear_active_backward_engine()) {}
IsolatedBackwardScope::~IsolatedBackwardScope() {
    _restore_active_backward_engine(previous_);
}

void Tape::add(std::shared_ptr<Node> node) { nodes_.push_back(std::move(node)); }
void Tape::backward(Tensor loss) { loss.backward(); }
void Tape::clear() { nodes_.clear(); }

} // namespace motifcl::autograd
