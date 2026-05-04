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
        default: throw Error("unknown dtype");
    }
}

std::size_t dtype_storage_nbytes(DType dtype, std::size_t numel) {
    if (dtype == DType::Q4_0) return (numel + 1) / 2;
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
        default: return "unknown";
    }
}

} // namespace motifcl
