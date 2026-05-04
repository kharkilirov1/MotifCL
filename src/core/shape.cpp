#include <motifcl/core/shape.hpp>
#include <motifcl/core/error.hpp>

#include <numeric>
#include <sstream>

namespace motifcl {

int64_t Shape::ndim() const {
    return static_cast<int64_t>(dims.size());
}

int64_t Shape::numel() const {
    if (dims.empty()) return 1;
    int64_t total = 1;
    for (auto d : dims) {
        MCL_CHECK(d >= 0, "shape dimensions must be non-negative");
        total *= d;
    }
    return total;
}

int64_t Shape::operator[](int index) const {
    if (index < 0) index += static_cast<int>(dims.size());
    MCL_CHECK(index >= 0 && static_cast<std::size_t>(index) < dims.size(), "shape index out of range");
    return dims[static_cast<std::size_t>(index)];
}

std::string Shape::str() const {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) oss << ", ";
        oss << dims[i];
    }
    oss << ']';
    return oss.str();
}

std::vector<int64_t> contiguous_strides(const Shape& shape) {
    std::vector<int64_t> strides(shape.dims.size(), 1);
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.dims.size()) - 1; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] = stride;
        stride *= shape.dims[static_cast<std::size_t>(i)];
    }
    return strides;
}

} // namespace motifcl
