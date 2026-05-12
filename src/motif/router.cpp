#include <motifcl/motif/router.hpp>

#include <cmath>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::motif {

Tensor router_soft(const Tensor& x, const Tensor& router_weight) {
    return softmax_rows(matmul(x, router_weight));
}

Tensor router_topk(const Tensor& x, const Tensor& router_weight, int k) {
    MCL_CHECK(k > 0, "router_topk expects k > 0");
    auto logits = matmul(x, router_weight);
    MCL_CHECK(logits.dtype() == DType::F32 && logits.ndim() == 2, "router_topk expects f32 [tokens, experts] logits");
    const int64_t rows = logits.shape()[0];
    const int64_t experts = logits.shape()[1];
    MCL_CHECK(k <= experts, "router_topk k exceeds expert count");

    auto host = logits.to_vector<float>();
    std::vector<float> out(host.size(), 0.0f);
    std::vector<int64_t> order(static_cast<std::size_t>(experts));
    for (int64_t r = 0; r < rows; ++r) {
        std::iota(order.begin(), order.end(), 0);
        const auto row_base = static_cast<std::size_t>(r * experts);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int64_t a, int64_t b) {
                              const float va = host[row_base + static_cast<std::size_t>(a)];
                              const float vb = host[row_base + static_cast<std::size_t>(b)];
                              if (va == vb) return a < b;
                              return va > vb;
                          });
        float maxv = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < k; ++i) {
            maxv = std::max(maxv, host[row_base + static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]);
        }
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            const auto e = order[static_cast<std::size_t>(i)];
            const float v = std::exp(host[row_base + static_cast<std::size_t>(e)] - maxv);
            out[row_base + static_cast<std::size_t>(e)] = v;
            sum += v;
        }
        MCL_CHECK(sum > 0.0f && std::isfinite(sum), "router_topk softmax normalization failed");
        for (int i = 0; i < k; ++i) {
            const auto e = order[static_cast<std::size_t>(i)];
            out[row_base + static_cast<std::size_t>(e)] /= sum;
        }
    }
    return Tensor::from_cpu(x.backend(), logits.shape(), DType::F32, out.data());
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
