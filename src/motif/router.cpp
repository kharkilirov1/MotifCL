#include <motifcl/motif/router.hpp>

#include <cmath>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::motif {

Tensor router_soft(const Tensor& x, const Tensor& router_weight) {
    return softmax_rows(matmul(x, router_weight));
}

Tensor router_topk(const Tensor& x, const Tensor& router_weight, int k) {
    (void)x;
    (void)router_weight;
    (void)k;
    throw Error("router_topk is intentionally not enabled yet; use router_soft for differentiable routing");
}

Router::Router(Backend& backend, int in_features, int motif_count_value, int top_k_value)
    : weight(Tensor::randn(backend, {in_features, motif_count_value}, 1.0f / std::sqrt(static_cast<float>(in_features)))),
      motif_count(motif_count_value),
      top_k(top_k_value) {}

Tensor Router::forward(const Tensor& x) {
    if (top_k > 0 && top_k < motif_count) return router_topk(x, weight.data, top_k);
    return router_soft(x, weight.data);
}

} // namespace motifcl::motif
