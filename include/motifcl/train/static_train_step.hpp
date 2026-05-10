#pragma once

#include <initializer_list>
#include <utility>
#include <vector>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/tensor/tensor.hpp>

namespace motifcl::train {

struct StaticTensorBinding {
    int captured_tensor_id = 0;
    Tensor tensor;
};

// Fixed-shape replay executor for an already-captured full training step.
// Intended capture body:
//   forward -> loss -> loss.backward() -> optimizer.step() -> optimizer.zero_grad()
// The replay path reuses the captured kernel sequence and buffers instead of
// rebuilding the autograd graph on each iteration.
class StaticTrainStepExecutor {
public:
    StaticTrainStepExecutor(autograd::CapturedGraph graph, Tensor loss);

    void execute();
    void bind_tensor(int captured_tensor_id, const Tensor& tensor);
    void bind_tensors(const std::vector<StaticTensorBinding>& bindings);
    void bind_tensors(std::initializer_list<StaticTensorBinding> bindings);
    void clear_bindings();
    void execute_with(const std::vector<StaticTensorBinding>& bindings);
    void execute_with(std::initializer_list<StaticTensorBinding> bindings);
    float execute_with_loss(const std::vector<StaticTensorBinding>& bindings);
    float execute_with_loss(std::initializer_list<StaticTensorBinding> bindings);

    Tensor loss() const { return loss_; }
    float loss_value() const { return loss_.item(); }
    const autograd::CapturedGraph& graph() const { return executor_.graph(); }
    const autograd::GraphRuntimePlan& runtime_plan() const { return executor_.runtime_plan(); }
    const std::string& execution_mode() const { return executor_.execution_mode(); }
    std::size_t executions() const { return executor_.executions(); }
    bool replayable() const { return executor_.replayable(); }
    bool rebindable() const { return executor_.rebindable(); }
    std::size_t bound_tensor_count() const { return executor_.bound_tensor_count(); }
    std::size_t arena_binding_count() const { return executor_.arena_binding_count(); }
    std::size_t arena_bytes() const { return executor_.arena_bytes(); }

private:
    autograd::GraphExecutor executor_;
    Tensor loss_;
};

template <typename CaptureFn>
StaticTrainStepExecutor capture_static_train_step(CaptureFn&& capture_fn) {
    autograd::GraphCaptureGuard guard;
    Tensor loss = capture_fn();
    auto graph = guard.finish();
    return StaticTrainStepExecutor(std::move(graph), std::move(loss));
}

} // namespace motifcl::train
