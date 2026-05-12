#include <motifcl/core/dtype.hpp>
#include <motifcl/core/error.hpp>

namespace motifcl {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::U8: return 1;
        case DType::Q8_0: return 1;
        case DType::Q4_0: return 1;
        case DType::Q4_K: return 1;
        case DType::Q5_K: return 1;
        case DType::Q6_K: return 1;
        case DType::Q4_0_COL: return 1;
        default: throw Error("unknown dtype");
    }
}

std::size_t dtype_storage_nbytes(DType dtype, std::size_t numel) {
    if (dtype == DType::Q4_0 || dtype == DType::Q4_0_COL) return (numel + 1) / 2;
    if (dtype == DType::Q4_K) {
        MCL_CHECK(numel % 256 == 0, "q4_k storage requires a multiple of 256 elements");
        return (numel / 256) * 144;
    }
    if (dtype == DType::Q5_K) {
        MCL_CHECK(numel % 256 == 0, "q5_k storage requires a multiple of 256 elements");
        return (numel / 256) * 176;
    }
    if (dtype == DType::Q6_K) {
        MCL_CHECK(numel % 256 == 0, "q6_k storage requires a multiple of 256 elements");
        return (numel / 256) * 210;
    }
    return numel * dtype_size(dtype);
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::I32: return "i32";
        case DType::U8: return "u8";
        case DType::Q8_0: return "q8_0";
        case DType::Q4_0: return "q4_0";
        case DType::Q4_K: return "q4_k";
        case DType::Q5_K: return "q5_k";
        case DType::Q6_K: return "q6_k";
        case DType::Q4_0_COL: return "q4_0_col";
        default: return "unknown";
    }
}

} // namespace motifcl
