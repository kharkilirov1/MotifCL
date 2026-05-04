#include <motifcl/motif/sarc_residual.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/reduce.hpp>
#include <motifcl/runtime/backend.hpp>

namespace motifcl::motif {

namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }
}

Tensor sarc_residual(const Tensor& x, const Tensor& fx, const Tensor& gamma, float eps) {
    MCL_CHECK(x.shape() == fx.shape(), "sarc_residual x/fx shape mismatch");
    MCL_CHECK(x.ndim() == 2 && gamma.ndim() == 1 && gamma.shape()[0] == x.shape()[1], "sarc_residual expects x/fx [rows, cols], gamma [cols]");
    auto out = Tensor::zeros(x.backend(), x.shape(), DType::F32);
    auto rms = rms_per_row(x, eps);
    auto k = x.backend().kernels.get("sarc_apply_f32");
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    int n = rows * cols;
    k.set_arg(0, x.buffer());
    k.set_arg(1, fx.buffer());
    k.set_arg(2, gamma.buffer());
    k.set_arg(3, rms.buffer());
    k.set_arg(4, out.buffer());
    k.set_arg(5, rows);
    k.set_arg(6, cols);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("sarc_apply_f32", {x.id(), fx.id(), gamma.id(), rms.id()}, {out.id()});
    return out;
}

SARCResidual::SARCResidual(Backend& backend, std::shared_ptr<nn::Module> branch_value, int features, float eps_value)
    : branch(std::move(branch_value)), gamma(Tensor::ones(backend, {features})), eps(eps_value) {}

Tensor SARCResidual::forward(const Tensor& x) {
    return sarc_residual(x, branch->forward(x), gamma.data, eps);
}

std::vector<nn::Parameter*> SARCResidual::parameters() {
    auto result = branch ? branch->parameters() : std::vector<nn::Parameter*>{};
    result.push_back(&gamma);
    return result;
}

} // namespace motifcl::motif
