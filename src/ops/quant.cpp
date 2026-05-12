#include <motifcl/ops/quant.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

namespace motifcl {
namespace {

constexpr std::size_t kLocal = 256;

std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

float choose_q8_scale(const Tensor& x) {
    auto host = x.to_vector<float>();
    float max_abs = 0.0f;
    for (float v : host) max_abs = std::max(max_abs, std::fabs(v));
    if (max_abs <= 0.0f) return 1.0f;
    return max_abs / 127.0f;
}

float choose_q4_scale(const Tensor& x) {
    auto host = x.to_vector<float>();
    float max_abs = 0.0f;
    for (float v : host) max_abs = std::max(max_abs, std::fabs(v));
    if (max_abs <= 0.0f) return 1.0f;
    return max_abs / 7.0f;
}

std::vector<float> choose_axis_scales(const Tensor& x, int axis, float qmax) {
    MCL_CHECK(x.ndim() == 2, "axis quantization expects rank-2 tensors");
    MCL_CHECK(axis == 0 || axis == 1, "axis quantization axis must be 0 (rows) or 1 (cols)");
    const int64_t rows = x.shape()[0];
    const int64_t cols = x.shape()[1];
    auto host = x.to_vector<float>();
    std::vector<float> scales(static_cast<std::size_t>(axis == 0 ? rows : cols), 1.0f);
    if (axis == 0) {
        for (int64_t r = 0; r < rows; ++r) {
            float max_abs = 0.0f;
            for (int64_t c = 0; c < cols; ++c) {
                max_abs = std::max(max_abs, std::fabs(host[static_cast<std::size_t>(r * cols + c)]));
            }
            scales[static_cast<std::size_t>(r)] = max_abs <= 0.0f ? 1.0f : max_abs / qmax;
        }
    } else {
        for (int64_t c = 0; c < cols; ++c) {
            float max_abs = 0.0f;
            for (int64_t r = 0; r < rows; ++r) {
                max_abs = std::max(max_abs, std::fabs(host[static_cast<std::size_t>(r * cols + c)]));
            }
            scales[static_cast<std::size_t>(c)] = max_abs <= 0.0f ? 1.0f : max_abs / qmax;
        }
    }
    return scales;
}

std::vector<float> choose_block_scales(const Tensor& x, int64_t block_size, float qmax) {
    MCL_CHECK(block_size > 0, "blockwise quantization block_size must be positive");
    auto host = x.to_vector<float>();
    const auto n = static_cast<std::size_t>(x.numel());
    const auto block = static_cast<std::size_t>(block_size);
    std::vector<float> scales((n + block - 1) / block, 1.0f);
    for (std::size_t bi = 0; bi < scales.size(); ++bi) {
        const std::size_t begin = bi * block;
        const std::size_t end = std::min(n, begin + block);
        float max_abs = 0.0f;
        for (std::size_t i = begin; i < end; ++i) max_abs = std::max(max_abs, std::fabs(host[i]));
        scales[bi] = max_abs <= 0.0f ? 1.0f : max_abs / qmax;
    }
    return scales;
}

Tensor make_scale_tensor(Backend& backend, const std::vector<float>& scales) {
    MCL_CHECK(!scales.empty(), "quant scale tensor must not be empty");
    return Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
}

int scale_mode_for_axis(int axis) {
    if (axis == 0) return 1;
    if (axis == 1) return 2;
    if (axis == 2) return 3;
    MCL_CHECK(false, "invalid quant scale axis");
    return 0;
}

Tensor quantize_scaled(const Tensor& x, DType dtype, const Tensor& scales, int axis, int64_t block_size) {
    MCL_CHECK(x.dtype() == DType::F32, "scaled quantization expects f32 input");
    MCL_CHECK(dtype == DType::Q8_0 || dtype == DType::Q4_0, "scaled quantization expects q8_0 or q4_0 output");
    MCL_CHECK(axis == 0 || axis == 1 || axis == 2, "scaled quantization axis must be 0, 1, or 2");
    MCL_CHECK(axis == 2 || x.ndim() == 2, "row/column quantization expects rank-2 tensors");

    auto out = Tensor::empty(x.backend(), x.shape(), dtype);
    out._set_quant_scales(scales, axis, block_size);
    auto k = x.backend().kernels.get(dtype == DType::Q8_0 ? "quantize_f32_to_q8_0_scaled" : "quantize_f32_to_q4_0_scaled");
    const int n = static_cast<int>(x.numel());
    const int rows = x.ndim() >= 1 ? static_cast<int>(x.shape()[0]) : 1;
    const int cols = x.ndim() >= 2 ? static_cast<int>(x.shape()[1]) : n;
    const int mode = scale_mode_for_axis(axis);
    const int block = static_cast<int>(block_size);
    const int work_items = dtype == DType::Q4_0
        ? static_cast<int>(dtype_storage_nbytes(DType::Q4_0, static_cast<std::size_t>(n)))
        : n;
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, scales.buffer());
    k.set_arg(3, n);
    k.set_arg(4, rows);
    k.set_arg(5, cols);
    k.set_arg(6, mode);
    k.set_arg(7, block);
    k.launch1d(round_up(static_cast<std::size_t>(work_items), kLocal), kLocal);
    autograd::record_op(dtype == DType::Q8_0 ? "quantize_f32_to_q8_0_scaled" : "quantize_f32_to_q4_0_scaled", {x.id(), scales.id()}, {out.id()});
    return out;
}

} // namespace

Tensor quantize_q8_symmetric(const Tensor& x, float scale) {
    MCL_CHECK(x.dtype() == DType::F32, "quantize_q8_symmetric expects f32 input");
    MCL_CHECK(x.valid(), "quantize_q8_symmetric input is invalid");
    if (scale <= 0.0f) scale = choose_q8_scale(x);
    MCL_CHECK(std::isfinite(scale) && scale > 0.0f, "quantize_q8_symmetric scale must be positive and finite");

    auto out = Tensor::empty(x.backend(), x.shape(), DType::Q8_0);
    out._set_quant_scale(scale);
    auto k = x.backend().kernels.get("quantize_f32_to_q8_0");
    int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, scale);
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op("quantize_f32_to_q8_0", {x.id()}, {out.id()});
    return out;
}

Tensor dequantize_q8(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::Q8_0, "dequantize_q8 expects q8_0 input");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    int n = static_cast<int>(x.numel());
    if (x.has_quant_scales()) {
        auto scales = x.quant_scales();
        auto k = x.backend().kernels.get("dequantize_q8_0_to_f32_scaled");
        const int rows = x.ndim() >= 1 ? static_cast<int>(x.shape()[0]) : 1;
        const int cols = x.ndim() >= 2 ? static_cast<int>(x.shape()[1]) : n;
        const int mode = scale_mode_for_axis(x.quant_scale_axis());
        const int block = static_cast<int>(x.quant_block_size());
        k.set_arg(0, x.buffer());
        k.set_arg(1, out.buffer());
        k.set_arg(2, scales.buffer());
        k.set_arg(3, n);
        k.set_arg(4, rows);
        k.set_arg(5, cols);
        k.set_arg(6, mode);
        k.set_arg(7, block);
        k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
        autograd::record_op("dequantize_q8_0_to_f32_scaled", {x.id(), scales.id()}, {out.id()});
        return out;
    }
    auto k = x.backend().kernels.get("dequantize_q8_0_to_f32");
    float scale = x.quant_scale();
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, scale);
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op("dequantize_q8_0_to_f32", {x.id()}, {out.id()});
    return out;
}

Tensor quantize_q4_symmetric(const Tensor& x, float scale) {
    MCL_CHECK(x.dtype() == DType::F32, "quantize_q4_symmetric expects f32 input");
    MCL_CHECK(x.valid(), "quantize_q4_symmetric input is invalid");
    if (scale <= 0.0f) scale = choose_q4_scale(x);
    MCL_CHECK(std::isfinite(scale) && scale > 0.0f, "quantize_q4_symmetric scale must be positive and finite");

    auto out = Tensor::empty(x.backend(), x.shape(), DType::Q4_0);
    out._set_quant_scale(scale);
    auto k = x.backend().kernels.get("quantize_f32_to_q4_0");
    int n = static_cast<int>(x.numel());
    int packed_n = static_cast<int>(dtype_storage_nbytes(DType::Q4_0, static_cast<std::size_t>(n)));
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, scale);
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(packed_n), kLocal), kLocal);
    autograd::record_op("quantize_f32_to_q4_0", {x.id()}, {out.id()});
    return out;
}

Tensor dequantize_q4(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::Q4_0, "dequantize_q4 expects q4_0 input");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    int n = static_cast<int>(x.numel());
    if (x.has_quant_scales()) {
        auto scales = x.quant_scales();
        auto k = x.backend().kernels.get("dequantize_q4_0_to_f32_scaled");
        const int rows = x.ndim() >= 1 ? static_cast<int>(x.shape()[0]) : 1;
        const int cols = x.ndim() >= 2 ? static_cast<int>(x.shape()[1]) : n;
        const int mode = scale_mode_for_axis(x.quant_scale_axis());
        const int block = static_cast<int>(x.quant_block_size());
        k.set_arg(0, x.buffer());
        k.set_arg(1, out.buffer());
        k.set_arg(2, scales.buffer());
        k.set_arg(3, n);
        k.set_arg(4, rows);
        k.set_arg(5, cols);
        k.set_arg(6, mode);
        k.set_arg(7, block);
        k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
        autograd::record_op("dequantize_q4_0_to_f32_scaled", {x.id(), scales.id()}, {out.id()});
        return out;
    }
    auto k = x.backend().kernels.get("dequantize_q4_0_to_f32");
    float scale = x.quant_scale();
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, scale);
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), kLocal), kLocal);
    autograd::record_op("dequantize_q4_0_to_f32", {x.id()}, {out.id()});
    return out;
}

Tensor quantize_q8_symmetric_axis(const Tensor& x, int axis) {
    auto scales = make_scale_tensor(x.backend(), choose_axis_scales(x, axis, 127.0f));
    return quantize_scaled(x, DType::Q8_0, scales, axis, 0);
}

Tensor quantize_q8_symmetric_rows(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "quantize_q8_symmetric_rows expects f32 input");
    MCL_CHECK(x.ndim() == 2, "row quantization expects rank-2 tensors");
    const auto rows = static_cast<int>(x.shape()[0]);
    const auto cols = static_cast<int>(x.shape()[1]);
    MCL_CHECK(rows > 0 && cols > 0, "row quantization expects non-empty rows and cols");
    auto scales = Tensor::empty(x.backend(), {rows}, DType::F32);
    auto out = Tensor::empty(x.backend(), x.shape(), DType::Q8_0);
    out._set_quant_scales(scales, 0, 0);
    auto k = x.backend().kernels.get("quantize_f32_to_q8_0_rowwise_fused");
    const int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, scales.buffer());
    k.set_arg(3, rows);
    k.set_arg(4, cols);
    k.set_arg(5, n);
    k.set_arg_local(6, kLocal * sizeof(float));
    k.launch1d(static_cast<std::size_t>(rows) * kLocal, kLocal);
    autograd::record_op("quantize_f32_to_q8_0_rowwise_fused", {x.id()}, {out.id()});
    return out;
}

Tensor quantize_q8_symmetric_cols(const Tensor& x) {
    return quantize_q8_symmetric_axis(x, 1);
}

Tensor quantize_q8_symmetric_blocks(const Tensor& x, int64_t block_size) {
    auto scales = make_scale_tensor(x.backend(), choose_block_scales(x, block_size, 127.0f));
    return quantize_scaled(x, DType::Q8_0, scales, 2, block_size);
}

Tensor quantize_q4_symmetric_axis(const Tensor& x, int axis) {
    auto scales = make_scale_tensor(x.backend(), choose_axis_scales(x, axis, 7.0f));
    return quantize_scaled(x, DType::Q4_0, scales, axis, 0);
}

Tensor quantize_q4_symmetric_rows(const Tensor& x) {
    return quantize_q4_symmetric_axis(x, 0);
}

Tensor quantize_q4_symmetric_cols(const Tensor& x) {
    return quantize_q4_symmetric_axis(x, 1);
}

Tensor quantize_q4_symmetric_blocks(const Tensor& x, int64_t block_size) {
    auto scales = make_scale_tensor(x.backend(), choose_block_scales(x, block_size, 7.0f));
    return quantize_scaled(x, DType::Q4_0, scales, 2, block_size);
}

} // namespace motifcl
