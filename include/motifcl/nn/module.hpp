#pragma once

#include <memory>
#include <vector>

#include <motifcl/nn/parameter.hpp>

namespace motifcl::nn {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Parameter*> parameters() = 0;

    virtual void zero_grad();
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {}; }
};

class GELU : public Module {
public:
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {}; }
};

class Sequential : public Module {
public:
    Sequential() = default;
    explicit Sequential(std::vector<std::shared_ptr<Module>> modules);

    void add(std::shared_ptr<Module> module);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace motifcl::nn
