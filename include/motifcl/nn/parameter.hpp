#pragma once

#include <motifcl/tensor/tensor.hpp>
#include <utility>

namespace motifcl::nn {

class Parameter {
public:
    Tensor data;
    bool trainable = true;

    Parameter() = default;
    explicit Parameter(Tensor tensor, bool is_trainable = true) : data(std::move(tensor)), trainable(is_trainable) {
        data.set_requires_grad(is_trainable);
    }

    std::shared_ptr<Tensor> grad() const { return data.grad(); }
    void zero_grad() { data.zero_grad(); }
};

} // namespace motifcl::nn
