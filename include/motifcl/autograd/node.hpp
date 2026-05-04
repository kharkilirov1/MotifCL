#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl::autograd {

class Node {
public:
    virtual ~Node() = default;
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
