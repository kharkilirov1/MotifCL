#include <motifcl/ops/reduce.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

#include <string>

namespace motifcl {

namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }
constexpr std::size_t kReduceWorkgroup = 256;
}

Tensor sum_rows(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "sum_rows supports f32 only");
    MCL_CHECK(x.ndim() == 2, "sum_rows expects rank-2 tensor");
    auto out = Tensor::empty(x.backend(), {x.shape()[1]}, DType::F32);
    auto k = x.backend().kernels.get("sum_rows_f32");
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    k.launch1d(round_up(static_cast<std::size_t>(cols), 256), 256);
    autograd::record_op("sum_rows_f32", {x.id()}, {out.id()});
    return out;
}

Tensor rowwise_sum(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "rowwise_sum supports f32 only");
    MCL_CHECK(x.ndim() == 2, "rowwise_sum expects rank-2 tensor");
    auto out = Tensor::empty(x.backend(), {x.shape()[0]}, DType::F32);
    const bool use_wg = x.backend().device_info().max_work_group_size >= kReduceWorkgroup;
    const std::string kernel_name = use_wg ? "rowwise_sum_wg_f32" : "rowwise_sum_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    if (use_wg) {
        k.launch2d(kReduceWorkgroup, static_cast<std::size_t>(rows), kReduceWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id()}, {out.id()});
    return out;
}

Tensor rowwise_max(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "rowwise_max supports f32 only");
    MCL_CHECK(x.ndim() == 2, "rowwise_max expects rank-2 tensor");
    auto out = Tensor::empty(x.backend(), {x.shape()[0]}, DType::F32);
    const bool use_wg = x.backend().device_info().max_work_group_size >= kReduceWorkgroup;
    const std::string kernel_name = use_wg ? "rowwise_max_wg_f32" : "rowwise_max_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    if (use_wg) {
        k.launch2d(kReduceWorkgroup, static_cast<std::size_t>(rows), kReduceWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id()}, {out.id()});
    return out;
}

Tensor rms_per_row(const Tensor& x, float eps) {
    MCL_CHECK(x.dtype() == DType::F32, "rms_per_row supports f32 only");
    MCL_CHECK(x.ndim() == 2, "rms_per_row expects rank-2 tensor");
    auto out = Tensor::empty(x.backend(), {x.shape()[0]}, DType::F32);
    const bool use_wg = x.backend().device_info().max_work_group_size >= kReduceWorkgroup;
    const std::string kernel_name = use_wg ? "rms_per_row_wg_f32" : "rms_per_row_f32";
    auto k = x.backend().kernels.get(kernel_name);
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    k.set_arg(4, eps);
    if (use_wg) {
        k.launch2d(kReduceWorkgroup, static_cast<std::size_t>(rows), kReduceWorkgroup, 1);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    }
    autograd::record_op(kernel_name, {x.id()}, {out.id()});
    return out;
}

} // namespace motifcl
