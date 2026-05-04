#include <motifcl/motif/motif_linear.hpp>
#include <motifcl/core/error.hpp>

#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::motif {

MotifLinear::MotifLinear(Backend& backend, int in_features, int out_features, int motif_count)
    : router(Tensor::randn(backend, {in_features, motif_count}, 0.02f)),
      in_features_(in_features), out_features_(out_features), motif_count_(motif_count) {
    motifs.reserve(static_cast<std::size_t>(motif_count));
    for (int i = 0; i < motif_count; ++i) {
        motifs.emplace_back(Tensor::randn(backend, {in_features, out_features}, 0.02f));
    }
}

Tensor MotifLinear::forward(const Tensor& x) {
    MCL_CHECK(motif_count_ > 0, "MotifLinear needs at least one motif");
    Tensor out = scale(matmul(x, motifs[0].data), 1.0f / static_cast<float>(motif_count_));
    for (int i = 1; i < motif_count_; ++i) {
        out = add(out, scale(matmul(x, motifs[static_cast<std::size_t>(i)].data), 1.0f / static_cast<float>(motif_count_)));
    }
    return out;
}

std::vector<nn::Parameter*> MotifLinear::parameters() {
    std::vector<nn::Parameter*> result;
    for (auto& m : motifs) result.push_back(&m);
    result.push_back(&router);
    return result;
}

} // namespace motifcl::motif
