#include <motifcl/motif/motif_block.hpp>

#include <motifcl/ops/basic_ops.hpp>

namespace motifcl::motif {

MotifTransformerBlock::MotifTransformerBlock(Backend& backend, int n_embd, int n_head, int mlp_hidden, bool use_sarc)
    : norm1_(backend, n_embd),
      attn_(backend, n_embd, n_head),
      norm2_(backend, n_embd),
      mlp_(backend, n_embd, mlp_hidden, n_embd),
      gamma_attn_(Tensor::ones(backend, {n_embd})),
      gamma_mlp_(Tensor::ones(backend, {n_embd})),
      use_sarc_(use_sarc) {}

Tensor MotifTransformerBlock::forward(const Tensor& x) {
    auto attn_out = attn_.forward(norm1_.forward(x), true);
    auto h = use_sarc_ ? sarc_residual(x, attn_out, gamma_attn_.data) : add(x, attn_out);
    auto mlp_out = mlp_.forward(norm2_.forward(h));
    return use_sarc_ ? sarc_residual(h, mlp_out, gamma_mlp_.data) : add(h, mlp_out);
}

std::vector<nn::Parameter*> MotifTransformerBlock::parameters() {
    std::vector<nn::Parameter*> result;
    for (auto* module : {static_cast<nn::Module*>(&norm1_), static_cast<nn::Module*>(&attn_), static_cast<nn::Module*>(&norm2_), static_cast<nn::Module*>(&mlp_)}) {
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    result.push_back(&gamma_attn_);
    result.push_back(&gamma_mlp_);
    return result;
}

} // namespace motifcl::motif
