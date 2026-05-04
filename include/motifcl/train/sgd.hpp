#pragma once

#include <motifcl/train/optimizer.hpp>

namespace motifcl::optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<nn::Parameter*> params, float lr = 1e-2f);
    void step() override;

private:
    float lr_ = 1e-2f;
};

} // namespace motifcl::optim
