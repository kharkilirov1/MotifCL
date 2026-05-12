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
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    // Q4_0 values repacked output-major: logical shape is [K, N], but
    // payload order is contiguous per output column: packed index = col*K + k.
    // Used for decode-time F32 x quantized-weight matmul.
    Q4_0_COL
};

std::size_t dtype_size(DType dtype);
std::size_t dtype_storage_nbytes(DType dtype, std::size_t numel);
const char* dtype_name(DType dtype);

} // namespace motifcl
