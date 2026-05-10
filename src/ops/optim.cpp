#include <motifcl/ops/optim.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cmath>

namespace motifcl {

namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }
}

void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    MCL_CHECK(param.dtype() == DType::F32 && grad.dtype() == DType::F32, "sgd_update supports f32 only");
    MCL_CHECK(param.shape() == grad.shape(), "sgd_update shape mismatch");
    auto k = param.backend().kernels.get("sgd_update_f32");
    int n = static_cast<int>(param.numel());
    k.set_arg(0, param.buffer());
    k.set_arg(1, grad.buffer());
    k.set_arg(2, lr);
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("sgd_update_f32", {param.id(), grad.id()}, {param.id()});
}

void adam_update(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                 float lr, float beta1, float beta2, float eps, int step) {
    adam_update_fast(param, grad, m, v, lr, beta1, beta2, eps, step);
}

void adam_update_fast(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                      float lr, float beta1, float beta2, float eps, int step, float weight_decay) {
    MCL_CHECK(param.dtype() == DType::F32 && grad.dtype() == DType::F32 && m.dtype() == DType::F32 && v.dtype() == DType::F32, "adam_update supports f32 only");
    MCL_CHECK(param.shape() == grad.shape() && param.shape() == m.shape() && param.shape() == v.shape(), "adam_update shape mismatch");
    MCL_CHECK(step > 0, "adam_update step must be positive");
    auto k = param.backend().kernels.get("adam_update_f32_fast");
    int n = static_cast<int>(param.numel());
    const float inv_bias1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
    const float inv_bias2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
    k.set_arg(0, param.buffer());
    k.set_arg(1, grad.buffer());
    k.set_arg(2, m.buffer());
    k.set_arg(3, v.buffer());
    k.set_arg(4, lr);
    k.set_arg(5, beta1);
    k.set_arg(6, beta2);
    k.set_arg(7, eps);
    k.set_arg(8, inv_bias1);
    k.set_arg(9, inv_bias2);
    k.set_arg(10, weight_decay);
    k.set_arg(11, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("adam_update_f32_fast", {param.id(), grad.id(), m.id(), v.id()}, {param.id(), m.id(), v.id()});
}

} // namespace motifcl
