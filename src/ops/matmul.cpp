#include <motifcl/ops/matmul.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/fp16.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cstdlib>
#include <memory>
#include <string>

namespace motifcl {

namespace {

std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

void launch_register_block4(Kernel& k, int M, int N) {
    constexpr std::size_t kBlock = 32;
    constexpr std::size_t kLocal = 8;
    k.launch2d((round_up(static_cast<std::size_t>(N), kBlock) / 4),
               (round_up(static_cast<std::size_t>(M), kBlock) / 4),
               kLocal,
               kLocal);
}

int preferred_f32_tile_from_env() {
    const char* env = std::getenv("MOTIFCL_MATMUL_F32_TILE");
    if (!env || !*env) return 0;
    const int tile = std::atoi(env);
    return (tile == 4 || tile == 8 || tile == 16) ? tile : 0;
}

bool shape_specific_matmul_dispatch_enabled() {
    const char* env = std::getenv("MOTIFCL_DISABLE_SHAPE_MATMUL_DISPATCH");
    return !(env && *env && std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "FALSE");
}

bool k512_matmul_enabled() {
    const char* env = std::getenv("MOTIFCL_DISABLE_K512_MATMUL");
    return !(env && *env && std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "FALSE");
}

int shape_specific_f32_tile(int M, int K, int N, bool trans_a, bool trans_b, const Backend& backend) {
    if (trans_a || trans_b || !shape_specific_matmul_dispatch_enabled()) return 0;
    if (K == 512 && (N == 512 || N == 256) && M >= 64 && M <= 512 &&
        backend.device_info().max_work_group_size >= 16 * 16) {
        return 16;
    }
    return 0;
}

int selected_f32_tile(int M, int K, int N, bool trans_a, bool trans_b, const Backend& backend) {
    const int forced = (!trans_a && !trans_b) ? preferred_f32_tile_from_env() : 0;
    if (forced > 0) return forced;
    return shape_specific_f32_tile(M, K, N, trans_a, trans_b, backend);
}

void launch_tiled(Kernel& k, int M, int N, int tile) {
    const auto t = static_cast<std::size_t>(tile);
    k.launch2d(round_up(static_cast<std::size_t>(N), t),
               round_up(static_cast<std::size_t>(M), t),
               t,
               t);
}

void require_matmul_f32(const Tensor& a, const Tensor& b) {
    MCL_CHECK(a.dtype() == DType::F32 && b.dtype() == DType::F32, "matmul supports f32 only");
    MCL_CHECK(a.ndim() == 2 && b.ndim() == 2, "matmul expects rank-2 tensors");
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), "matmul requires tensors on same backend");
}

void require_matmul_q8(const Tensor& a, const Tensor& b) {
    MCL_CHECK(a.dtype() == DType::Q8_0 && b.dtype() == DType::Q8_0, "q8 matmul expects q8_0 tensors");
    MCL_CHECK(a.ndim() == 2 && b.ndim() == 2, "q8 matmul expects rank-2 tensors");
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), "q8 matmul requires tensors on same backend");
}

void require_matmul_q4(const Tensor& a, const Tensor& b) {
    MCL_CHECK(a.dtype() == DType::Q4_0 && b.dtype() == DType::Q4_0, "q4 matmul expects q4_0 tensors");
    MCL_CHECK(a.ndim() == 2 && b.ndim() == 2, "q4 matmul expects rank-2 tensors");
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), "q4 matmul requires tensors on same backend");
}

bool is_quant_dtype(DType dtype) {
    return dtype == DType::Q8_0 || dtype == DType::Q4_0;
}

void require_matmul_quantized(const Tensor& a, const Tensor& b) {
    MCL_CHECK(is_quant_dtype(a.dtype()) && is_quant_dtype(b.dtype()), "quantized matmul expects q8_0 or q4_0 tensors");
    MCL_CHECK(a.ndim() == 2 && b.ndim() == 2, "quantized matmul expects rank-2 tensors");
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), "quantized matmul requires tensors on same backend");
    MCL_CHECK(a.shape()[1] == b.shape()[0], "quantized matmul inner dimension mismatch");
}

int scale_mode(const Tensor& x) {
    if (!x.has_quant_scales()) return 0;
    if (x.quant_scale_axis() == 0) return 1;
    if (x.quant_scale_axis() == 1) return 2;
    if (x.quant_scale_axis() == 2) return 3;
    MCL_CHECK(false, "invalid quant scale axis");
    return 0;
}

std::string quant_scaled_kernel_name(const Tensor& a, const Tensor& b) {
    if (a.dtype() == DType::Q8_0 && b.dtype() == DType::Q8_0) return "matmul_q8_0_scaled_f32";
    if (a.dtype() == DType::Q4_0 && b.dtype() == DType::Q4_0) return "matmul_q4_0_scaled_f32";
    if (a.dtype() == DType::Q8_0 && b.dtype() == DType::Q4_0) return "matmul_q8_q4_scaled_f32";
    if (a.dtype() == DType::Q4_0 && b.dtype() == DType::Q8_0) return "matmul_q4_q8_scaled_f32";
    MCL_CHECK(false, "unsupported quantized matmul dtype combination");
    return {};
}

Tensor matmul_q8(const Tensor& a, const Tensor& b) {
    require_matmul_q8(a, b);
    MCL_CHECK(a.shape()[1] == b.shape()[0], "q8 matmul inner dimension mismatch");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    auto& backend = a.backend();
    const std::string dot_mode = backend.int_dot_mode();
    const bool use_khr_int_dot = (K % 4 == 0) && dot_mode == "cl_khr_integer_dot_product";
    const std::string kernel_name = use_khr_int_dot ? "matmul_q8_0_khr_dot_f32" : "matmul_q8_0_rb4_f32";
    auto k = use_khr_int_dot
        ? backend.kernels.get_q8_int_dot_variant(dot_mode)
        : backend.kernels.get(kernel_name);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    k.set_arg(6, a.quant_scale());
    k.set_arg(7, b.quant_scale());
    if (use_khr_int_dot) {
        k.launch2d(round_up(static_cast<std::size_t>(N), 16), round_up(static_cast<std::size_t>(M), 16), 16, 16);
    } else {
        launch_register_block4(k, M, N);
    }
    autograd::record_op(kernel_name, {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor matmul_q4(const Tensor& a, const Tensor& b) {
    require_matmul_q4(a, b);
    MCL_CHECK(a.shape()[1] == b.shape()[0], "q4 matmul inner dimension mismatch");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    const bool use_dot4 = (K % 4 == 0);
    const std::string kernel_name = use_dot4 ? "matmul_q4_0_dot4_rb4_f32" : "matmul_q4_0_rb4_f32";
    auto k = a.backend().kernels.get(kernel_name);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    k.set_arg(6, a.quant_scale());
    k.set_arg(7, b.quant_scale());
    launch_register_block4(k, M, N);
    autograd::record_op(kernel_name, {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor matmul_q8_q4(const Tensor& a, const Tensor& b) {
    require_matmul_quantized(a, b);
    MCL_CHECK(a.dtype() == DType::Q8_0 && b.dtype() == DType::Q4_0, "q8/q4 matmul expects q8_0 lhs and q4_0 rhs");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    auto k = a.backend().kernels.get("matmul_q8_q4_rb4_f32");
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    k.set_arg(6, a.quant_scale());
    k.set_arg(7, b.quant_scale());
    launch_register_block4(k, M, N);
    autograd::record_op("matmul_q8_q4_rb4_f32", {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor matmul_q4_q8(const Tensor& a, const Tensor& b) {
    require_matmul_quantized(a, b);
    MCL_CHECK(a.dtype() == DType::Q4_0 && b.dtype() == DType::Q8_0, "q4/q8 matmul expects q4_0 lhs and q8_0 rhs");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    auto k = a.backend().kernels.get("matmul_q4_q8_rb4_f32");
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    k.set_arg(6, a.quant_scale());
    k.set_arg(7, b.quant_scale());
    launch_register_block4(k, M, N);
    autograd::record_op("matmul_q4_q8_rb4_f32", {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor matmul_quant_scaled(const Tensor& a, const Tensor& b) {
    require_matmul_quantized(a, b);
    MCL_CHECK(a.has_quant_scales() || b.has_quant_scales(), "scaled quantized matmul requires at least one scale tensor");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    auto k = a.backend().kernels.get(quant_scaled_kernel_name(a, b));
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    Tensor fallback_scales = a.has_quant_scales() ? a.quant_scales() : b.quant_scales();
    Tensor scales_a = a.has_quant_scales() ? a.quant_scales() : fallback_scales;
    Tensor scales_b = b.has_quant_scales() ? b.quant_scales() : fallback_scales;
    int mode_a = scale_mode(a);
    int mode_b = scale_mode(b);
    int block_a = static_cast<int>(a.quant_block_size());
    int block_b = static_cast<int>(b.quant_block_size());
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    k.set_arg(6, a.quant_scale());
    k.set_arg(7, b.quant_scale());
    k.set_arg(8, scales_a.buffer());
    k.set_arg(9, scales_b.buffer());
    k.set_arg(10, mode_a);
    k.set_arg(11, mode_b);
    k.set_arg(12, block_a);
    k.set_arg(13, block_b);
    k.launch2d(round_up(static_cast<std::size_t>(N), 16), round_up(static_cast<std::size_t>(M), 16), 16, 16);
    autograd::record_op(quant_scaled_kernel_name(a, b), {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor matmul_flags(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b) {
    require_matmul_f32(a, b);
    int64_t a_m = trans_a ? a.shape()[1] : a.shape()[0];
    int64_t a_k = trans_a ? a.shape()[0] : a.shape()[1];
    int64_t b_k = trans_b ? b.shape()[1] : b.shape()[0];
    int64_t b_n = trans_b ? b.shape()[0] : b.shape()[1];
    MCL_CHECK(a_k == b_k, "matmul inner dimension mismatch");
    auto out = Tensor::empty(a.backend(), {a_m, b_n}, DType::F32);
    int M = static_cast<int>(a_m);
    int N = static_cast<int>(b_n);
    int K = static_cast<int>(a_k);
    const int preferred_tile = selected_f32_tile(M, K, N, trans_a, trans_b, a.backend());
    if (preferred_tile > 0) {
        MCL_CHECK(static_cast<std::size_t>(preferred_tile * preferred_tile) <= a.backend().device_info().max_work_group_size,
                  "selected matmul tile exceeds device max work-group size");
        auto k = a.backend().kernels.get_matmul_tiled_variant(preferred_tile);
        k.set_arg(0, a.buffer());
        k.set_arg(1, b.buffer());
        k.set_arg(2, out.buffer());
        k.set_arg(3, M);
        k.set_arg(4, N);
        k.set_arg(5, K);
        launch_tiled(k, M, N, preferred_tile);
        autograd::record_op("matmul_tuned_f32_t" + std::to_string(preferred_tile), {a.id(), b.id()}, {out.id()});
        return out;
    }
    std::string kernel_name;
    const bool use_k512 = K == 512 && k512_matmul_enabled();
    if (use_k512 && !trans_a && !trans_b) {
        kernel_name = "matmul_register_block4_k512_f32";
    } else if (use_k512 && trans_a && !trans_b) {
        kernel_name = "matmul_transa_rb4_k512_f32";
    } else if (use_k512 && !trans_a && trans_b) {
        kernel_name = "matmul_transb_rb4_k512_f32";
    } else if (!trans_a && !trans_b) {
        kernel_name = "matmul_register_block4_f32";
    } else if (trans_a && !trans_b) {
        kernel_name = "matmul_transa_rb4_f32";
    } else if (!trans_a && trans_b) {
        kernel_name = "matmul_transb_rb4_f32";
    } else {
        kernel_name = "matmul_flags_f32";
    }
    auto k = a.backend().kernels.get(kernel_name);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    if (kernel_name == "matmul_flags_f32") {
        int ta = trans_a ? 1 : 0;
        int tb = trans_b ? 1 : 0;
        k.set_arg(6, ta);
        k.set_arg(7, tb);
        k.launch2d(round_up(static_cast<std::size_t>(N), 16), round_up(static_cast<std::size_t>(M), 16), 16, 16);
    } else {
        launch_register_block4(k, M, N);
    }
    autograd::record_op(kernel_name, {a.id(), b.id()}, {out.id()});
    return out;
}

struct MatMulBackward : autograd::Node {
    Tensor a, b;
    MatMulBackward(Tensor a, Tensor b) : a(std::move(a)), b(std::move(b)) {}
    std::vector<Tensor> inputs() const override { return {a, b}; }
    void backward(const Tensor& grad_output) override {
        if (a.requires_grad()) a.backward(matmul_transpose_b(grad_output, b));
        if (b.requires_grad()) b.backward(matmul_transpose_a(a, grad_output));
    }
};

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dtype() == DType::F16 || b.dtype() == DType::F16) {
        MCL_CHECK(a.dtype() == DType::F16 && b.dtype() == DType::F16, "f16 matmul expects both inputs to be f16");
        MCL_CHECK(!a.requires_grad() && !b.requires_grad(), "f16 matmul autograd is not implemented; use f32 training path for now");
        return matmul_f16_accum_f32(a, b);
    }
    const bool quantized = a.dtype() == DType::Q8_0 || b.dtype() == DType::Q8_0 ||
                           a.dtype() == DType::Q4_0 || b.dtype() == DType::Q4_0;
    if (quantized) {
        MCL_CHECK(!a.requires_grad() && !b.requires_grad(), "quantized matmul does not support autograd");
        require_matmul_quantized(a, b);
        if (a.has_quant_scales() || b.has_quant_scales()) return matmul_quant_scaled(a, b);
        if (a.dtype() == DType::Q8_0 && b.dtype() == DType::Q8_0) return matmul_q8(a, b);
        if (a.dtype() == DType::Q4_0 && b.dtype() == DType::Q4_0) return matmul_q4(a, b);
        if (a.dtype() == DType::Q8_0 && b.dtype() == DType::Q4_0) return matmul_q8_q4(a, b);
        if (a.dtype() == DType::Q4_0 && b.dtype() == DType::Q8_0) return matmul_q4_q8(a, b);
        MCL_CHECK(false, "unsupported quantized matmul dtype combination");
    }
    auto out = matmul_flags(a, b, false, false);
    if (autograd::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MatMulBackward>(a, b));
    }
    return out;
}

Tensor matmul_transpose_a(const Tensor& a, const Tensor& b) {
    return matmul_flags(a, b, true, false);
}

Tensor matmul_transpose_b(const Tensor& a, const Tensor& b) {
    return matmul_flags(a, b, false, true);
}

void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) {
    require_matmul_f32(a, b);
    MCL_CHECK(out.dtype() == DType::F32, "matmul_out supports f32 output only");
    MCL_CHECK(a.shape()[1] == b.shape()[0], "matmul_out inner dimension mismatch");
    MCL_CHECK(out.shape() == Shape({a.shape()[0], b.shape()[1]}), "matmul_out output shape mismatch");
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    const int preferred_tile = selected_f32_tile(M, K, N, false, false, a.backend());
    if (preferred_tile > 0) {
        auto k = a.backend().kernels.get_matmul_tiled_variant(preferred_tile);
        k.set_arg(0, a.buffer());
        k.set_arg(1, b.buffer());
        k.set_arg(2, out.buffer());
        k.set_arg(3, M);
        k.set_arg(4, N);
        k.set_arg(5, K);
        launch_tiled(k, M, N, preferred_tile);
        autograd::record_op("matmul_tuned_f32_t" + std::to_string(preferred_tile), {a.id(), b.id()}, {out.id()});
        return;
    }
    const std::string kernel_name = (K == 512 && k512_matmul_enabled())
        ? "matmul_register_block4_k512_f32"
        : "matmul_register_block4_f32";
    auto k = a.backend().kernels.get(kernel_name);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    launch_register_block4(k, M, N);
    autograd::record_op(kernel_name, {a.id(), b.id()}, {out.id()});
}

Tensor matmul_tiled_variant(const Tensor& a, const Tensor& b, int tile) {
    require_matmul_f32(a, b);
    MCL_CHECK(a.shape()[1] == b.shape()[0], "matmul_tiled_variant inner dimension mismatch");
    MCL_CHECK(tile == 4 || tile == 8 || tile == 16, "matmul_tiled_variant tile must be one of 4, 8, or 16");
    MCL_CHECK(static_cast<std::size_t>(tile * tile) <= a.backend().device_info().max_work_group_size,
              "matmul_tiled_variant tile exceeds device max work-group size");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    auto k = a.backend().kernels.get_matmul_tiled_variant(tile);
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    launch_tiled(k, M, N, tile);
    autograd::record_op("matmul_tiled_f32_t" + std::to_string(tile), {a.id(), b.id()}, {out.id()});
    return out;
}

Tensor matmul_f16_accum_f32(const Tensor& a, const Tensor& b) {
    MCL_CHECK(a.dtype() == DType::F16 && b.dtype() == DType::F16, "matmul_f16_accum_f32 expects f16 inputs");
    MCL_CHECK(a.ndim() == 2 && b.ndim() == 2, "matmul_f16_accum_f32 expects rank-2 tensors");
    MCL_CHECK(a.backend_ptr() == b.backend_ptr(), "matmul_f16_accum_f32 requires tensors on same backend");
    MCL_CHECK(a.shape()[1] == b.shape()[0], "matmul_f16_accum_f32 inner dimension mismatch");
    MCL_CHECK(backend_supports_fp16(a.backend()), "backend does not expose cl_khr_fp16");
    auto out = Tensor::empty(a.backend(), {a.shape()[0], b.shape()[1]}, DType::F32);
    int M = static_cast<int>(a.shape()[0]);
    int N = static_cast<int>(b.shape()[1]);
    int K = static_cast<int>(a.shape()[1]);
    const bool use_vec4 = (K % 4) == 0;
    const char* kernel_name = use_vec4 ? "matmul_f16_accum_f32_vec4" : "matmul_f16_accum_f32";
    auto k = a.backend().kernels.get(kernel_name);
    k.set_arg(0, a.buffer());
    k.set_arg(1, b.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, M);
    k.set_arg(4, N);
    k.set_arg(5, K);
    k.launch2d(round_up(static_cast<std::size_t>(N), 16), round_up(static_cast<std::size_t>(M), 16), 16, 16);
    autograd::record_op(kernel_name, {a.id(), b.id()}, {out.id()});
    return out;
}

} // namespace motifcl
