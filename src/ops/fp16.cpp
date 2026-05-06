#include <motifcl/ops/fp16.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

#include <string>

namespace motifcl {

namespace {

constexpr std::size_t kLocal = 256;

std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

bool contains(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}

} // namespace

bool backend_supports_fp16(const Backend& backend) {
    return contains(backend.device_info().extensions, "cl_khr_fp16");
}

Tensor cast_f32_to_f16(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "cast_f32_to_f16 expects f32 input");
    MCL_CHECK(backend_supports_fp16(x.backend()), "backend does not expose cl_khr_fp16");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F16);
    auto k = x.backend().kernels.get("cast_f32_to_f16");
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op("cast_f32_to_f16", {x.id()}, {out.id()});
    return out;
}

Tensor cast_f16_to_f32(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F16, "cast_f16_to_f32 expects f16 input");
    MCL_CHECK(backend_supports_fp16(x.backend()), "backend does not expose cl_khr_fp16");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get("cast_f16_to_f32");
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op("cast_f16_to_f32", {x.id()}, {out.id()});
    return out;
}

} // namespace motifcl
