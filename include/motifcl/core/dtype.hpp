#pragma once

#include <cstddef>
#include <string>

namespace motifcl {

enum class DType {
    F32,
    F16,
    I32,
    U8,
    Q8_0,
    Q4_0
};

std::size_t dtype_size(DType dtype);
std::size_t dtype_storage_nbytes(DType dtype, std::size_t numel);
const char* dtype_name(DType dtype);

} // namespace motifcl
