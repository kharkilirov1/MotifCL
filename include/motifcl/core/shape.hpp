#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace motifcl {

class Shape {
public:
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> values) : dims(values) {}
    explicit Shape(std::vector<int64_t> values) : dims(std::move(values)) {}

    int64_t ndim() const;
    int64_t numel() const;
    int64_t operator[](int index) const;
    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return !(*this == other); }
    std::string str() const;
};

std::vector<int64_t> contiguous_strides(const Shape& shape);

} // namespace motifcl
