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

} // namespace motifcl::autograd
