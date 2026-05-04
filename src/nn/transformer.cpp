#include <motifcl/nn/transformer.hpp>

#include <cmath>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::nn {

Tensor MLP::forward(const Tensor& x) {
    return fc2.forward(motifcl::gelu(fc1.forward(x)));
}

std::vector<Parameter*> MLP::parameters() {
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}

TransformerBlock::TransformerBlock(Backend& backend, int n_embd, int n_head, int mlp_hidden)
    : norm1_(backend, n_embd), attn_(backend, n_embd, n_head), norm2_(backend, n_embd), mlp_(backend, n_embd, mlp_hidden, n_embd) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0]);
}

Tensor TransformerBlock::forward(const Tensor& x, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(x.ndim() == 2, "TransformerBlock expects flattened [tokens, channels] input");
    MCL_CHECK(batch_size > 0 && seq_len > 0 && batch_size * seq_len == x.shape()[0], "TransformerBlock invalid batch/sequence dimensions");
    auto h = add(x, attn_.forward(norm1_.forward(x), batch_size, seq_len, true));
    return add(h, mlp_.forward(norm2_.forward(h)));
}

std::vector<Parameter*> TransformerBlock::parameters() {
    std::vector<Parameter*> result;
    for (auto* module : {static_cast<Module*>(&norm1_), static_cast<Module*>(&attn_), static_cast<Module*>(&norm2_), static_cast<Module*>(&mlp_)}) {
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

GPTModel::GPTModel(Backend& backend, int vocab_size, int block_size, int n_embd, int n_head, int n_layer, int mlp_hidden)
    : token_embedding(Tensor::randn(backend, {vocab_size, n_embd}, 0.02f)),
      position_embedding(Tensor::randn(backend, {block_size, n_embd}, 0.02f)),
      final_norm(backend, n_embd),
      lm_head(Tensor::randn(backend, {n_embd, vocab_size}, 0.02f)),
      vocab_size_(vocab_size),
      block_size_(block_size),
      n_embd_(n_embd) {
    blocks.reserve(static_cast<std::size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        blocks.push_back(std::make_shared<TransformerBlock>(backend, n_embd, n_head, mlp_hidden));
    }
}

Tensor GPTModel::forward(const Tensor& token_ids) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "GPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "GPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T <= block_size_, "sequence length exceeds GPTModel block_size");

    Tensor h = token_position_embedding(token_ids, token_embedding.data, position_embedding.data); // [B*T, C]
    for (auto& block : blocks) h = block->forward(h, B, T);
    h = final_norm.forward(h);
    Tensor logits = matmul(h, lm_head.data); // [B*T, V]
    if (token_ids.ndim() == 2) return logits.view({B, T, vocab_size_});
    return logits;
}

std::vector<Parameter*> GPTModel::parameters() {
    std::vector<Parameter*> result{&token_embedding, &position_embedding};
    for (auto& block : blocks) {
        auto p = block->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto nf = final_norm.parameters();
    result.insert(result.end(), nf.begin(), nf.end());
    result.push_back(&lm_head);
    return result;
}

} // namespace motifcl::nn
