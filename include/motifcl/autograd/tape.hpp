#pragma once

#include <memory>
#include <vector>

#include <motifcl/autograd/node.hpp>

namespace motifcl::autograd {

class Tape {
public:
    void add(std::shared_ptr<Node> node);
    void backward(Tensor loss);
    void clear();
    std::size_t size() const { return nodes_.size(); }

private:
    std::vector<std::shared_ptr<Node>> nodes_;
};

} // namespace motifcl::autograd
