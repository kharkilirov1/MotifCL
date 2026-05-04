#include <motifcl/motif/motif_lora.hpp>

#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::motif {

MotifLoRA::MotifLoRA(Backend& backend, int in_features, int out_features, int rank_value, int motif_count_value, float alpha_value)
    : frozen_weight(Tensor::randn(backend, {in_features, out_features}, 0.02f)),
      router(Tensor::randn(backend, {in_features, motif_count_value}, 0.02f)),
      rank(rank_value), motif_count(motif_count_value), alpha(alpha_value) {
    frozen_weight.set_requires_grad(false);
    for (int i = 0; i < motif_count; ++i) {
        lora_A.emplace_back(Tensor::randn(backend, {in_features, rank}, 0.02f));
        lora_B.emplace_back(Tensor::zeros(backend, {rank, out_features}));
    }
}

Tensor MotifLoRA::forward(const Tensor& x) {
    Tensor out = matmul(x, frozen_weight);
    float scale_value = alpha / static_cast<float>(rank > 0 ? rank : 1) / static_cast<float>(motif_count > 0 ? motif_count : 1);
    for (int i = 0; i < motif_count; ++i) {
        auto delta = matmul(matmul(x, lora_A[static_cast<std::size_t>(i)].data), lora_B[static_cast<std::size_t>(i)].data);
        out = add(out, scale(delta, scale_value));
    }
    return out;
}

std::vector<nn::Parameter*> MotifLoRA::parameters() {
    std::vector<nn::Parameter*> result;
    for (auto& p : lora_A) result.push_back(&p);
    for (auto& p : lora_B) result.push_back(&p);
    result.push_back(&router);
    return result;
}

} // namespace motifcl::motif
