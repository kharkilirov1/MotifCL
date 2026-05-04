#include <motifcl/nn/module.hpp>

#include <motifcl/ops/activation.hpp>

namespace motifcl::nn {

void Module::zero_grad() {
    for (auto* p : parameters()) {
        if (p) p->zero_grad();
    }
}

Tensor ReLU::forward(const Tensor& x) { return motifcl::relu(x); }
Tensor GELU::forward(const Tensor& x) { return motifcl::gelu(x); }

Sequential::Sequential(std::vector<std::shared_ptr<Module>> modules) : modules_(std::move(modules)) {}

void Sequential::add(std::shared_ptr<Module> module) { modules_.push_back(std::move(module)); }

Tensor Sequential::forward(const Tensor& x) {
    Tensor out = x;
    for (auto& m : modules_) out = m->forward(out);
    return out;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> result;
    for (auto& m : modules_) {
        auto p = m->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

} // namespace motifcl::nn
