#pragma once

#include <motifcl/motif/sarc_residual.hpp>
#include <motifcl/nn/attention.hpp>
#include <motifcl/nn/mlp.hpp>
#include <motifcl/nn/rmsnorm.hpp>

namespace motifcl::motif {

class MotifTransformerBlock : public nn::Module {
public:
    MotifTransformerBlock(Backend& backend, int n_embd, int n_head, int mlp_hidden, bool use_sarc = true);
    Tensor forward(const Tensor& x) override;
    std::vector<nn::Parameter*> parameters() override;

private:
    nn::RMSNorm norm1_;
    nn::SelfAttention attn_;
    nn::RMSNorm norm2_;
    nn::MLP mlp_;
    nn::Parameter gamma_attn_;
    nn::Parameter gamma_mlp_;
    bool use_sarc_ = true;
};

} // namespace motifcl::motif
