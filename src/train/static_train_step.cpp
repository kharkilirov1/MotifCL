#include <motifcl/train/static_train_step.hpp>

#include <utility>

namespace motifcl::train {

StaticTrainStepExecutor::StaticTrainStepExecutor(autograd::CapturedGraph graph, Tensor loss)
    : executor_(std::move(graph)), loss_(std::move(loss)) {}

void StaticTrainStepExecutor::execute() {
    executor_.execute();
}

void StaticTrainStepExecutor::bind_tensor(int captured_tensor_id, const Tensor& tensor) {
    executor_.bind_tensor(captured_tensor_id, tensor);
}

void StaticTrainStepExecutor::bind_tensors(const std::vector<StaticTensorBinding>& bindings) {
    for (const auto& binding : bindings) {
        bind_tensor(binding.captured_tensor_id, binding.tensor);
    }
}

void StaticTrainStepExecutor::bind_tensors(std::initializer_list<StaticTensorBinding> bindings) {
    for (const auto& binding : bindings) {
        bind_tensor(binding.captured_tensor_id, binding.tensor);
    }
}

void StaticTrainStepExecutor::clear_bindings() {
    executor_.clear_bindings();
}

void StaticTrainStepExecutor::execute_with(const std::vector<StaticTensorBinding>& bindings) {
    clear_bindings();
    bind_tensors(bindings);
    execute();
}

void StaticTrainStepExecutor::execute_with(std::initializer_list<StaticTensorBinding> bindings) {
    clear_bindings();
    bind_tensors(bindings);
    execute();
}

float StaticTrainStepExecutor::execute_with_loss(const std::vector<StaticTensorBinding>& bindings) {
    execute_with(bindings);
    return loss_value();
}

float StaticTrainStepExecutor::execute_with_loss(std::initializer_list<StaticTensorBinding> bindings) {
    execute_with(bindings);
    return loss_value();
}

} // namespace motifcl::train
