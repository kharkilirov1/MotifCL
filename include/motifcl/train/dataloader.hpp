#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl::train {

struct Batch {
    Tensor x;
    Tensor y;
};

class DataLoader {
public:
    virtual ~DataLoader() = default;
    virtual Batch next() = 0;
};

} // namespace motifcl::train
