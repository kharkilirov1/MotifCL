#pragma once

#include <cstdint>
#include <algorithm>
#include <utility>

#include <motifcl/ops/indexing.hpp>
#include <motifcl/tensor/tensor.hpp>

namespace motifcl::train {

struct Batch {
    Tensor x;
    Tensor y;
};

class IDataLoader {
public:
    virtual ~IDataLoader() = default;
    virtual Batch next() = 0;
};

class TensorDataLoader : public IDataLoader {
public:
    TensorDataLoader(Tensor x, Tensor y, int64_t batch_size, bool drop_last = false)
        : x_(std::move(x)), y_(std::move(y)), batch_size_(batch_size), drop_last_(drop_last) {}

    Batch next() override {
        const int64_t rows = x_.shape()[0];
        if (cursor_ >= rows) cursor_ = 0;
        int64_t end = cursor_ + batch_size_;
        if (end > rows) {
            if (drop_last_) {
                cursor_ = 0;
                end = std::min(batch_size_, rows);
            } else {
                end = rows;
            }
        }
        Batch batch{slice_rows(x_, cursor_, end), slice_rows(y_, cursor_, end)};
        cursor_ = end >= rows ? 0 : end;
        return batch;
    }

private:
    Tensor x_;
    Tensor y_;
    int64_t batch_size_ = 1;
    bool drop_last_ = false;
    int64_t cursor_ = 0;
};

} // namespace motifcl::train
