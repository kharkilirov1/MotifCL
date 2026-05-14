#include <motifcl/nn/transformer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/fused_transformer.hpp>
#include <motifcl/ops/indexing.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/quant.hpp>
#include <motifcl/runtime/backend.hpp>

namespace motifcl::nn {

namespace {

Parameter random_parameter_or_empty(Backend& backend,
                                    const std::vector<int64_t>& shape,
                                    float scale,
                                    bool skip_weight_init) {
    if (skip_weight_init) {
        Parameter p;
        p.trainable = false;
        return p;
    }
    return Parameter(Tensor::randn(backend, Shape(shape), scale));
}

Tensor last_sequence_hidden_rows(const Tensor& h, int64_t batch_size, int64_t seq_len, int64_t width) {
    MCL_CHECK(h.dtype() == DType::F32 && h.ndim() == 2, "last hidden rows expects flattened f32 [B*T,H]");
    MCL_CHECK(batch_size > 0 && seq_len > 0 && width > 0, "last hidden rows invalid dimensions");
    MCL_CHECK(h.shape()[0] == batch_size * seq_len && h.shape()[1] == width,
              "last hidden rows shape mismatch");
    if (seq_len == 1) return h.view({batch_size, width});
    std::vector<int32_t> positions(static_cast<std::size_t>(batch_size),
                                   static_cast<int32_t>(seq_len - 1));
    auto pos = Tensor::from_cpu(h.backend(), {batch_size}, DType::I32, positions.data());
    return gather_last_token_logits(h, pos, batch_size, seq_len, width);
}

void check_decode_step_token_shape(const Tensor& token_ids, const char* owner) {
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, std::string(owner) + " decode_step expects [1] or [B,1]");
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T == 1, std::string(owner) + " decode_step expects exactly one token per batch");
}

} // namespace

Tensor MLP::forward(const Tensor& x) {
    auto h = motifcl::matmul(x, fc1.weight.data);
    h = fc1.has_bias() ? motifcl::add_bias_gelu_rows(h, fc1.bias.data) : motifcl::gelu(h);
    return fc2.forward(h);
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

namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

TransformerConfig normalize_config(TransformerConfig cfg) {
    MCL_CHECK(cfg.vocab_size > 0, "TransformerConfig vocab_size must be positive");
    MCL_CHECK(cfg.block_size > 0, "TransformerConfig block_size must be positive");
    MCL_CHECK(cfg.n_embd > 0, "TransformerConfig n_embd must be positive");
    MCL_CHECK(cfg.n_head > 0, "TransformerConfig n_head must be positive");
    MCL_CHECK(cfg.n_layer > 0, "TransformerConfig n_layer must be positive");
    if (cfg.n_kv_head <= 0) cfg.n_kv_head = cfg.n_head;
    MCL_CHECK(cfg.n_head % cfg.n_kv_head == 0, "TransformerConfig n_head must be divisible by n_kv_head");
    if (cfg.head_dim <= 0) {
        MCL_CHECK(cfg.n_embd % cfg.n_head == 0, "TransformerConfig n_embd must be divisible by n_head when head_dim is unset");
        cfg.head_dim = cfg.n_embd / cfg.n_head;
    }
    MCL_CHECK(cfg.head_dim > 0, "TransformerConfig head_dim must be positive");
    if (!cfg.layer_head_dims.empty()) {
        MCL_CHECK(static_cast<int>(cfg.layer_head_dims.size()) == cfg.n_layer,
                  "TransformerConfig layer_head_dims must match n_layer");
        for (int value : cfg.layer_head_dims) {
            MCL_CHECK(value > 0, "TransformerConfig layer_head_dims entries must be positive");
        }
    }
    if (cfg.rotary_dim <= 0) cfg.rotary_dim = cfg.head_dim;
    const int max_head_dim = cfg.layer_head_dims.empty()
        ? cfg.head_dim
        : *std::max_element(cfg.layer_head_dims.begin(), cfg.layer_head_dims.end());
    MCL_CHECK(cfg.rotary_dim <= max_head_dim && cfg.rotary_dim % 2 == 0,
              "TransformerConfig rotary_dim must be even and <= head_dim/max(layer_head_dims)");
    if (cfg.mlp_hidden <= 0) cfg.mlp_hidden = cfg.use_swiglu ? (cfg.n_embd * 8 / 3) : (cfg.n_embd * 4);
    if (!cfg.layer_mlp_hiddens.empty()) {
        MCL_CHECK(static_cast<int>(cfg.layer_mlp_hiddens.size()) == cfg.n_layer,
                  "TransformerConfig layer_mlp_hiddens must match n_layer");
        for (int value : cfg.layer_mlp_hiddens) {
            MCL_CHECK(value > 0, "TransformerConfig layer_mlp_hiddens entries must be positive");
        }
    }
    if (!cfg.layer_rope_thetas.empty()) {
        MCL_CHECK(static_cast<int>(cfg.layer_rope_thetas.size()) == cfg.n_layer,
                  "TransformerConfig layer_rope_thetas must match n_layer");
        for (float value : cfg.layer_rope_thetas) {
            MCL_CHECK(value > 0.0f, "TransformerConfig layer_rope_thetas entries must be positive");
        }
    }
    MCL_CHECK(cfg.dropout >= 0.0f && cfg.dropout < 1.0f, "TransformerConfig dropout must be in [0, 1)");
    MCL_CHECK(cfg.rms_norm_eps > 0.0f, "TransformerConfig rms_norm_eps must be positive");
    MCL_CHECK(cfg.sliding_window >= 0, "TransformerConfig sliding_window must be non-negative");
    MCL_CHECK(cfg.embedding_scale > 0.0f, "TransformerConfig embedding_scale must be positive");
    if (cfg.use_per_layer_inputs) {
        MCL_CHECK(cfg.per_layer_input_dim > 0, "TransformerConfig per_layer_input_dim must be positive when PLE is enabled");
        if (cfg.per_layer_input_vocab_size <= 0) cfg.per_layer_input_vocab_size = cfg.vocab_size;
        MCL_CHECK(cfg.per_layer_input_vocab_size > 0,
                  "TransformerConfig per_layer_input_vocab_size must be positive when PLE is enabled");
    }
    return cfg;
}

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return std::string(value) != "0" && std::string(value) != "false" && std::string(value) != "FALSE";
}

bool decode_block_graph_replay_enabled() {
    return !env_enabled("MOTIFCL_DISABLE_DECODE_BLOCK_GRAPH_REPLAY");
}

bool decode_block_graph_replay_log_enabled() {
    return env_enabled("MOTIFCL_LOG_DECODE_BLOCK_GRAPH_REPLAY");
}

bool decode_full_block_graph_replay_enabled() {
    return env_enabled("MOTIFCL_ENABLE_DECODE_FULL_BLOCK_GRAPH_REPLAY") &&
           !env_enabled("MOTIFCL_DISABLE_DECODE_BLOCK_GRAPH_REPLAY") &&
           !env_enabled("MOTIFCL_DISABLE_DECODE_FULL_BLOCK_GRAPH_REPLAY");
}

struct DecodeBlockTailGraphCache {
    bool capture_failed = false;
    bool ready = false;
    int input_id = 0;
    int packed_id = 0;
    int output_id = 0;
    Shape input_shape;
    Shape packed_shape;
    Shape output_shape;
    DType output_dtype = DType::F32;
    int layer_index = -1;
    std::size_t node_count = 0;
    std::size_t kernel_count = 0;
    std::unique_ptr<autograd::GraphExecutor> executor;
};

std::mutex g_decode_block_tail_graph_mutex;
std::unordered_map<const ModernTransformerBlock*, DecodeBlockTailGraphCache> g_decode_block_tail_graph_cache;

struct DecodeFullBlockGraphCache {
    bool capture_failed = false;
    bool ready = false;
    int input_id = 0;
    int packed_id = 0;
    int cache_k_id = 0;
    int cache_v_id = 0;
    int output_id = 0;
    Shape input_shape;
    Shape packed_shape;
    Shape cache_k_shape;
    Shape cache_v_shape;
    Shape output_shape;
    DType output_dtype = DType::F32;
    int layer_index = -1;
    int64_t offset = -1;
    int64_t batch_size = 0;
    int64_t seq_len = 0;
    int64_t token_count = 0;
    std::size_t node_count = 0;
    std::size_t kernel_count = 0;
    std::unique_ptr<autograd::GraphExecutor> executor;
};

std::mutex g_decode_full_block_graph_mutex;
std::unordered_map<const ModernTransformerBlock*, std::unordered_map<std::string, DecodeFullBlockGraphCache>>
    g_decode_full_block_graph_cache;

bool kquant_m1_wg4_enabled() {
    return !env_enabled("MOTIFCL_DISABLE_KQUANT_M1_WG4");
}

bool q4_0_col_wg64x4_enabled() {
    return !env_enabled("MOTIFCL_DISABLE_Q4_0_COL_WG64X4");
}

bool q4_0_tile8_wg64x8_enabled() {
    return !env_enabled("MOTIFCL_DISABLE_Q4_0_TILE8_WG64X8");
}

bool can_use_fused_swiglu_mlp_residual(const ModernMLP& mlp) {
    return env_enabled("MOTIFCL_ENABLE_FUSED_MLP_BACKWARD") &&
           !mlp.split_projections_enabled() &&
           mlp.use_swiglu &&
           mlp.dropout_p == 0.0f &&
           !mlp.gate_up_proj.has_bias() &&
           !mlp.down_proj.has_bias();
}

Tensor fused_or_eager_mlp_residual(const Tensor& h, RMSNorm& norm, ModernMLP& mlp) {
    if (can_use_fused_swiglu_mlp_residual(mlp)) {
        return motifcl::fused_swiglu_mlp_rmsnorm_residual(
            h,
            norm.weight.data,
            mlp.gate_up_proj.weight.data,
            mlp.down_proj.weight.data,
            norm.eps);
    }
    return motifcl::add(h, mlp.forward(norm.forward(h)));
}

bool is_k_quant_dtype(DType dtype) {
    return dtype == DType::Q4_K || dtype == DType::Q5_K || dtype == DType::Q6_K;
}

bool is_direct_decode_quant_dtype(DType dtype) {
    return dtype == DType::Q4_0_COL || is_k_quant_dtype(dtype);
}

const char* fused_decode_pair_kernel_name(DType dtype, bool use_wg) {
    if (dtype == DType::Q4_K) return use_wg ? "matmul_f32_q4_k_m1_2out_wg_f32" : "matmul_f32_q4_k_m1_2out_f32";
    if (dtype == DType::Q5_K) return use_wg ? "matmul_f32_q5_k_m1_2out_wg_f32" : "matmul_f32_q5_k_m1_2out_f32";
    if (dtype == DType::Q6_K) return use_wg ? "matmul_f32_q6_k_m1_2out_wg_f32" : "matmul_f32_q6_k_m1_2out_f32";
    MCL_CHECK(false, "unsupported K-quant dtype for fused decode pair");
    return "";
}

const char* fused_decode_pair_wg4_kernel_name(DType dtype) {
    if (dtype == DType::Q4_K) return "matmul_f32_q4_k_m1_2out_wg4_f32";
    if (dtype == DType::Q4_0_COL) return "matmul_f32_q4_0_col_m1_2out_wg4_f32";
    MCL_CHECK(false, "unsupported quant dtype for fused decode pair wg4");
    return "";
}

const char* fused_decode_triple_kernel_name(DType dtype, bool use_wg) {
    if (dtype == DType::Q4_K) return use_wg ? "matmul_f32_q4_k_m1_3out_wg_f32" : "matmul_f32_q4_k_m1_3out_f32";
    if (dtype == DType::Q5_K) return use_wg ? "matmul_f32_q5_k_m1_3out_wg_f32" : "matmul_f32_q5_k_m1_3out_f32";
    if (dtype == DType::Q6_K) return use_wg ? "matmul_f32_q6_k_m1_3out_wg_f32" : "matmul_f32_q6_k_m1_3out_f32";
    MCL_CHECK(false, "unsupported K-quant dtype for fused decode triple");
    return "";
}

const char* fused_decode_triple_wg4_kernel_name(DType dtype) {
    if (dtype == DType::Q4_K) return "matmul_f32_q4_k_m1_3out_wg4_f32";
    if (dtype == DType::Q4_0_COL) return "matmul_f32_q4_0_col_m1_3out_wg4_f32";
    MCL_CHECK(false, "unsupported quant dtype for fused decode triple wg4");
    return "";
}

int q4_0_col_block_size(const Tensor& weight) {
    MCL_CHECK(weight.dtype() == DType::Q4_0_COL && weight.has_quant_scales() &&
                  (weight.quant_scale_axis() == 3 || weight.quant_scale_axis() == 4) &&
                  weight.quant_block_size() > 0,
              "Q4_0_COL fused decode expects column-block scales");
    return static_cast<int>(weight.quant_block_size());
}

bool q4_0_col_tile8_layout(const Tensor& weight) {
    return weight.valid() && weight.dtype() == DType::Q4_0_COL && weight.quant_scale_axis() == 4;
}

const Tensor& decode_quantized_weight(const Linear& layer) {
    return layer.decode_quantized_weight().valid() ? layer.decode_quantized_weight() : layer.quantized_weight();
}

DType decode_quantized_weight_dtype(const Linear& layer) {
    return layer.decode_quantized_weight().valid()
        ? layer.decode_quantized_weight_dtype()
        : layer.quantized_weight_dtype();
}

bool can_use_fused_silu_down_q4_0_col(const Tensor& gate, const Tensor& up, const Linear& down) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_SILU_DOWN_Q4_0_COL") || autograd::is_enabled()) {
        return false;
    }
    if (!gate.valid() || !up.valid() || gate.dtype() != DType::F32 || up.dtype() != DType::F32) return false;
    if (gate.ndim() != 2 || up.ndim() != 2 || gate.shape() != up.shape() || gate.shape()[0] != 1) return false;
    const Tensor& down_weight = decode_quantized_weight(down);
    if (!down.quantized_inference_enabled() || decode_quantized_weight_dtype(down) != DType::Q4_0_COL ||
        !down_weight.valid() || down.has_bias()) {
        return false;
    }
    if (down_weight.shape()[0] != gate.shape()[1] ||
        down_weight.shape()[1] != down.out_features()) {
        return false;
    }
    if (gate.backend_ptr() != up.backend_ptr() || gate.backend_ptr() != down_weight.backend_ptr()) return false;
    if (gate.backend().device_info().max_work_group_size < 64) return false;
    if (!down_weight.has_quant_scales()) return false;
    const int axis = down_weight.quant_scale_axis();
    if (axis == 3 && !env_enabled("MOTIFCL_ENABLE_FUSED_SILU_DOWN_Q4_0_COL")) return false;
    return axis == 3 || axis == 4;
}

Tensor fused_silu_down_q4_0_col(const Tensor& gate, const Tensor& up, const Linear& down) {
    MCL_CHECK(can_use_fused_silu_down_q4_0_col(gate, up, down),
              "fused_silu_down_q4_0_col expects one-token SwiGLU input and Q4_0_COL down weight");
    auto out = Tensor::empty(gate.backend(), {1, down.out_features()}, DType::F32);
    const Tensor& down_weight = decode_quantized_weight(down);
    const bool tile8 = down_weight.quant_scale_axis() == 4;
    const char* kernel_name = tile8
        ? "matmul_silu_product_q4_0_tile8_m1_wg64x8_f32"
        : "matmul_silu_product_q4_0_col_m1_wg64x4_f32";
    auto kernel = gate.backend().kernels.get(kernel_name);
    const int n = down.out_features();
    const int kdim = static_cast<int>(gate.shape()[1]);
    const int block = q4_0_col_block_size(down_weight);
    MCL_CHECK(!tile8 || (kdim % block) == 0,
              "Q4_0_COL tile8 fused SwiGLU down requires K divisible by block size");
    const int blocks_per_col = (kdim + block - 1) / block;
    auto scales = down_weight.quant_scales();
    constexpr std::size_t kLocal = 64;
    kernel.set_arg(0, gate.buffer());
    kernel.set_arg(1, up.buffer());
    kernel.set_arg(2, down_weight.buffer());
    kernel.set_arg(3, scales.buffer());
    kernel.set_arg(4, out.buffer());
    kernel.set_arg(5, n);
    kernel.set_arg(6, kdim);
    kernel.set_arg(7, blocks_per_col);
    kernel.set_arg(8, block);
    kernel.set_arg_local(9, (tile8 ? 8 : 4) * kLocal * sizeof(float));
    const std::size_t groups = tile8
        ? (static_cast<std::size_t>(n) + 7u) / 8u
        : (static_cast<std::size_t>(n) + 3u) / 4u;
    kernel.launch1d(groups * kLocal, kLocal);
    autograd::record_op(kernel_name,
                        {gate.id(), up.id(), down_weight.id(), scales.id()},
                        {out.id()});
    return out;
}

bool can_use_fused_decode_projection(const Tensor& x, const Linear& layer) {
    const Tensor& qweight = decode_quantized_weight(layer);
    const DType qdtype = decode_quantized_weight_dtype(layer);
    return x.valid() &&
           x.dtype() == DType::F32 &&
           x.ndim() == 2 &&
           x.shape()[0] == 1 &&
           layer.quantized_inference_enabled() &&
           is_direct_decode_quant_dtype(qdtype) &&
           qweight.valid() &&
           qweight.ndim() == 2 &&
           qweight.shape()[0] == x.shape()[1] &&
           qweight.shape()[1] == layer.out_features() &&
           !layer.has_bias() &&
           (!autograd::is_enabled() || !layer.weight.data.valid()) &&
           x.backend_ptr() == qweight.backend_ptr();
}

bool can_use_fused_decode_projection_pair(const Tensor& x, const Linear& a, const Linear& b) {
    return can_use_fused_decode_projection(x, a) &&
           can_use_fused_decode_projection(x, b) &&
           decode_quantized_weight_dtype(a) == decode_quantized_weight_dtype(b);
}

bool can_use_fused_decode_swiglu_product_pair(const Tensor& x, const Linear& gate, const Linear& up) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_SWIGLU_PRODUCT_PAIR") || autograd::is_enabled()) return false;
    if (!can_use_fused_decode_projection_pair(x, gate, up)) return false;
    if (decode_quantized_weight_dtype(gate) != DType::Q4_0_COL) return false;
    const Tensor& gate_weight = decode_quantized_weight(gate);
    const Tensor& up_weight = decode_quantized_weight(up);
    if (!q4_0_col_tile8_layout(gate_weight) || !q4_0_col_tile8_layout(up_weight)) return false;
    if (gate.out_features() != up.out_features()) return false;
    const int block = q4_0_col_block_size(gate_weight);
    if (q4_0_col_block_size(up_weight) != block) return false;
    if (gate_weight.quant_scale_axis() != up_weight.quant_scale_axis()) return false;
    if ((static_cast<int>(x.shape()[1]) % block) != 0) return false;
    return x.backend().device_info().max_work_group_size >= 64;
}

Tensor fused_decode_swiglu_product_pair(const Tensor& x, const Linear& gate, const Linear& up) {
    MCL_CHECK(can_use_fused_decode_swiglu_product_pair(x, gate, up),
              "fused_decode_swiglu_product_pair expects compatible M=1 F32 input and Q4_0_COL tile8 gate/up weights");
    auto out = Tensor::empty(x.backend(), {1, gate.out_features()}, DType::F32);
    const Tensor& gate_weight = decode_quantized_weight(gate);
    const Tensor& up_weight = decode_quantized_weight(up);
    const int in = static_cast<int>(x.shape()[1]);
    const int n = gate.out_features();
    const int block = q4_0_col_block_size(gate_weight);
    const int blocks_per_col = (in + block - 1) / block;
    auto gate_scales = gate_weight.quant_scales();
    auto up_scales = up_weight.quant_scales();
    auto kernel = x.backend().kernels.get("matmul_swiglu_product_f32_q4_0_tile8_m1_wg64x8_f32");
    constexpr std::size_t kLocal = 64;
    kernel.set_arg(0, x.buffer());
    kernel.set_arg(1, gate_weight.buffer());
    kernel.set_arg(2, gate_scales.buffer());
    kernel.set_arg(3, up_weight.buffer());
    kernel.set_arg(4, up_scales.buffer());
    kernel.set_arg(5, out.buffer());
    kernel.set_arg(6, n);
    kernel.set_arg(7, in);
    kernel.set_arg(8, blocks_per_col);
    kernel.set_arg(9, block);
    kernel.set_arg_local(10, 16 * kLocal * sizeof(float));
    const std::size_t groups = (static_cast<std::size_t>(n) + 7u) / 8u;
    kernel.launch1d(groups * kLocal, kLocal);
    autograd::record_op("matmul_swiglu_product_f32_q4_0_tile8_m1_wg64x8_f32",
                        {x.id(), gate_weight.id(), gate_scales.id(), up_weight.id(), up_scales.id()},
                        {out.id()});
    return out;
}

bool can_use_fused_decode_projection_triple(const Tensor& x,
                                            const Linear& a,
                                            const Linear& b,
                                            const Linear& c) {
    return can_use_fused_decode_projection_pair(x, a, b) &&
           can_use_fused_decode_projection(x, c) &&
           decode_quantized_weight_dtype(a) == decode_quantized_weight_dtype(c);
}

std::pair<Tensor, Tensor> fused_decode_projection_pair(const Tensor& x,
                                                       const Linear& a,
                                                       const Linear& b) {
    MCL_CHECK(can_use_fused_decode_projection_pair(x, a, b),
              "fused decode projection pair expects compatible M=1 F32 input and K-quant weights");
    auto out_a = Tensor::empty(x.backend(), {1, a.out_features()}, DType::F32);
    auto out_b = Tensor::empty(x.backend(), {1, b.out_features()}, DType::F32);
    const int n0 = a.out_features();
    const int n1 = b.out_features();
    const int in = static_cast<int>(x.shape()[1]);
    const Tensor& weight_a = decode_quantized_weight(a);
    const Tensor& weight_b = decode_quantized_weight(b);
    const DType weight_dtype = decode_quantized_weight_dtype(a);
    constexpr int kLocal = 128;
    const int total = n0 + n1;
    const bool can_wg = in >= kLocal &&
                        x.backend().device_info().max_work_group_size >= static_cast<std::size_t>(kLocal) &&
                        !env_enabled("MOTIFCL_DISABLE_KQUANT_M1_WG");
    const bool use_wg4 = can_wg &&
                         (weight_dtype == DType::Q4_0_COL ||
                          (weight_dtype == DType::Q4_K &&
                           env_enabled("MOTIFCL_ENABLE_KQUANT_PAIR_WG4"))) &&
                         total <= 32768;
    const bool use_wg = !use_wg4 && total <= 32768 && can_wg && env_enabled("MOTIFCL_ENABLE_KQUANT_PAIR_WG");
    const bool q4_tile8 = weight_dtype == DType::Q4_0_COL &&
                          q4_0_col_tile8_layout(weight_a) &&
                          q4_0_col_tile8_layout(weight_b);
    const bool use_q4col_wg64x4 = use_wg4 &&
                                  weight_dtype == DType::Q4_0_COL &&
                                  !q4_tile8 &&
                                  q4_0_col_wg64x4_enabled() &&
                                  x.backend().device_info().max_work_group_size >= 64;
    const bool use_q4tile8_wg64x8 = use_wg4 && q4_tile8 &&
                                    q4_0_tile8_wg64x8_enabled() &&
                                    x.backend().device_info().max_work_group_size >= 64;
    const char* kernel_name = use_q4tile8_wg64x8
        ? "matmul_f32_q4_0_tile8_m1_2out_wg64x8_f32"
        : (use_q4col_wg64x4
        ? "matmul_f32_q4_0_col_m1_2out_wg64x4_f32"
        : (use_wg4 ? fused_decode_pair_wg4_kernel_name(weight_dtype)
                   : fused_decode_pair_kernel_name(weight_dtype, use_wg)));
    auto k = x.backend().kernels.get(kernel_name);
    std::vector<int> graph_inputs{x.id(), weight_a.id(), weight_b.id()};
    if (weight_dtype == DType::Q4_0_COL) {
        MCL_CHECK(use_wg4, "Q4_0_COL fused decode pair requires wg4 path");
        const int block = q4_0_col_block_size(weight_a);
        MCL_CHECK(q4_0_col_block_size(weight_b) == block,
                  "Q4_0_COL fused decode pair expects matching block sizes");
        MCL_CHECK(weight_a.quant_scale_axis() == weight_b.quant_scale_axis(),
                  "Q4_0_COL fused decode pair expects matching layouts");
        MCL_CHECK(!q4_tile8 || (in % block) == 0,
                  "Q4_0_COL tile8 fused decode pair requires K divisible by block size");
        const int blocks_per_col = (in + block - 1) / block;
        auto scales_a = weight_a.quant_scales();
        auto scales_b = weight_b.quant_scales();
        graph_inputs = {x.id(),
                        weight_a.id(), scales_a.id(),
                        weight_b.id(), scales_b.id()};
        k.set_arg(0, x.buffer());
        k.set_arg(1, weight_a.buffer());
        k.set_arg(2, scales_a.buffer());
        k.set_arg(3, weight_b.buffer());
        k.set_arg(4, scales_b.buffer());
        k.set_arg(5, out_a.buffer());
        k.set_arg(6, out_b.buffer());
        k.set_arg(7, n0);
        k.set_arg(8, n1);
        k.set_arg(9, in);
        k.set_arg(10, blocks_per_col);
        k.set_arg(11, block);
        const int local = (use_q4tile8_wg64x8 || use_q4col_wg64x4) ? 64 : kLocal;
        k.set_arg_local(12, (use_q4tile8_wg64x8 ? 8 * local : (use_q4col_wg64x4 ? 4 * local : local)) * sizeof(float));
        const std::size_t groups = use_q4tile8_wg64x8 ? (static_cast<std::size_t>(total) + 7u) / 8u
                                                       : (static_cast<std::size_t>(total) + 3u) / 4u;
        k.launch1d(groups * local, local);
    } else {
        k.set_arg(0, x.buffer());
        k.set_arg(1, weight_a.buffer());
        k.set_arg(2, weight_b.buffer());
        k.set_arg(3, out_a.buffer());
        k.set_arg(4, out_b.buffer());
        k.set_arg(5, n0);
        k.set_arg(6, n1);
        k.set_arg(7, in);
        if (use_wg4) {
            k.set_arg_local(8, kLocal * sizeof(float));
            const std::size_t groups = (static_cast<std::size_t>(total) + 3u) / 4u;
            k.launch1d(groups * kLocal, kLocal);
        } else if (use_wg) {
            k.set_arg_local(8, kLocal * sizeof(float));
            k.launch1d(static_cast<std::size_t>(total) * kLocal, kLocal);
        } else {
            k.launch1d(round_up(static_cast<std::size_t>(total), 128), 128);
        }
    }
    autograd::record_op(kernel_name, std::move(graph_inputs), {out_a.id(), out_b.id()});
    return {out_a, out_b};
}


bool can_use_fused_decode_projection_pair_rmsnorm(const Tensor& x,
                                                  const RMSNorm& norm,
                                                  const Linear& a,
                                                  const Linear& b) {
    if (!env_enabled("MOTIFCL_ENABLE_FUSED_RMSNORM_DECODE_PAIR") ||
        env_enabled("MOTIFCL_DISABLE_FUSED_RMSNORM_DECODE_PAIR") ||
        autograd::is_enabled()) return false;
    if (!can_use_fused_decode_projection_pair(x, a, b)) return false;
    if (!norm.weight.data.valid() || norm.weight.data.dtype() != DType::F32 || norm.weight.data.ndim() != 1) return false;
    if (norm.weight.data.shape()[0] != x.shape()[1]) return false;
    if (x.backend_ptr() != norm.weight.data.backend_ptr()) return false;
    const Tensor& weight_a = decode_quantized_weight(a);
    const Tensor& weight_b = decode_quantized_weight(b);
    if (decode_quantized_weight_dtype(a) != DType::Q4_0_COL) return false;
    if (!q4_0_col_tile8_layout(weight_a) || !q4_0_col_tile8_layout(weight_b)) return false;
    const int block = q4_0_col_block_size(weight_a);
    if (q4_0_col_block_size(weight_b) != block) return false;
    if (weight_a.quant_scale_axis() != weight_b.quant_scale_axis()) return false;
    if ((static_cast<int>(x.shape()[1]) % block) != 0) return false;
    return x.backend().device_info().max_work_group_size >= 64;
}

std::pair<Tensor, Tensor> fused_decode_projection_pair_rmsnorm(const Tensor& x,
                                                               const RMSNorm& norm,
                                                               const Linear& a,
                                                               const Linear& b) {
    MCL_CHECK(can_use_fused_decode_projection_pair_rmsnorm(x, norm, a, b),
              "fused decode RMSNorm projection pair expects compatible M=1 F32 input and Q4_0_COL tile8 weights");
    auto out_a = Tensor::empty(x.backend(), {1, a.out_features()}, DType::F32);
    auto out_b = Tensor::empty(x.backend(), {1, b.out_features()}, DType::F32);
    auto kernel = x.backend().kernels.get("matmul_rmsnorm_f32_q4_0_tile8_m1_2out_wg64x8_f32");
    const int n0 = a.out_features();
    const int n1 = b.out_features();
    const int in = static_cast<int>(x.shape()[1]);
    const Tensor& weight_a = decode_quantized_weight(a);
    const Tensor& weight_b = decode_quantized_weight(b);
    const int block = q4_0_col_block_size(weight_a);
    const int blocks_per_col = (in + block - 1) / block;
    auto scales_a = weight_a.quant_scales();
    auto scales_b = weight_b.quant_scales();
    constexpr std::size_t kLocal = 64;
    kernel.set_arg(0, x.buffer());
    kernel.set_arg(1, norm.weight.data.buffer());
    kernel.set_arg(2, weight_a.buffer());
    kernel.set_arg(3, scales_a.buffer());
    kernel.set_arg(4, weight_b.buffer());
    kernel.set_arg(5, scales_b.buffer());
    kernel.set_arg(6, out_a.buffer());
    kernel.set_arg(7, out_b.buffer());
    kernel.set_arg(8, n0);
    kernel.set_arg(9, n1);
    kernel.set_arg(10, in);
    kernel.set_arg(11, blocks_per_col);
    kernel.set_arg(12, block);
    kernel.set_arg(13, norm.eps);
    kernel.set_arg_local(14, 8 * kLocal * sizeof(float));
    const std::size_t groups = (static_cast<std::size_t>(n0 + n1) + 7u) / 8u;
    kernel.launch1d(groups * kLocal, kLocal);
    autograd::record_op("matmul_rmsnorm_f32_q4_0_tile8_m1_2out_wg64x8_f32",
                        {x.id(), norm.weight.data.id(),
                         weight_a.id(), scales_a.id(),
                         weight_b.id(), scales_b.id()},
                        {out_a.id(), out_b.id()});
    return {out_a, out_b};
}

QKV fused_decode_projection_triple(const Tensor& x,
                                   const Linear& q,
                                   const Linear& k_proj,
                                   const Linear& v) {
    MCL_CHECK(can_use_fused_decode_projection_triple(x, q, k_proj, v),
              "fused decode projection triple expects compatible M=1 F32 input and K-quant weights");
    QKV out{
        Tensor::empty(x.backend(), {1, q.out_features()}, DType::F32),
        Tensor::empty(x.backend(), {1, k_proj.out_features()}, DType::F32),
        Tensor::empty(x.backend(), {1, v.out_features()}, DType::F32)};
    const int n0 = q.out_features();
    const int n1 = k_proj.out_features();
    const int n2 = v.out_features();
    const int in = static_cast<int>(x.shape()[1]);
    const Tensor& weight_q = decode_quantized_weight(q);
    const Tensor& weight_k = decode_quantized_weight(k_proj);
    const Tensor& weight_v = decode_quantized_weight(v);
    const DType weight_dtype = decode_quantized_weight_dtype(q);
    constexpr int kLocal = 128;
    const int total = n0 + n1 + n2;
    const bool can_wg = total <= 32768 &&
                        in >= kLocal &&
                        x.backend().device_info().max_work_group_size >= static_cast<std::size_t>(kLocal) &&
                        !env_enabled("MOTIFCL_DISABLE_KQUANT_M1_WG");
    const bool use_wg4 = can_wg && kquant_m1_wg4_enabled() &&
                         (weight_dtype == DType::Q4_K ||
                          weight_dtype == DType::Q4_0_COL);
    const bool use_wg = can_wg && !use_wg4;
    const bool q4_tile8 = weight_dtype == DType::Q4_0_COL &&
                          q4_0_col_tile8_layout(weight_q) &&
                          q4_0_col_tile8_layout(weight_k) &&
                          q4_0_col_tile8_layout(weight_v);
    const bool use_q4col_wg64x4 = use_wg4 &&
                                  weight_dtype == DType::Q4_0_COL &&
                                  !q4_tile8 &&
                                  q4_0_col_wg64x4_enabled() &&
                                  x.backend().device_info().max_work_group_size >= 64;
    const bool use_q4tile8_wg64x8 = use_wg4 && q4_tile8 &&
                                    q4_0_tile8_wg64x8_enabled() &&
                                    x.backend().device_info().max_work_group_size >= 64;
    const char* kernel_name = use_q4tile8_wg64x8
        ? "matmul_f32_q4_0_tile8_m1_3out_wg64x8_f32"
        : (use_q4col_wg64x4
        ? "matmul_f32_q4_0_col_m1_3out_wg64x4_f32"
        : (use_wg4 ? fused_decode_triple_wg4_kernel_name(weight_dtype)
                   : fused_decode_triple_kernel_name(weight_dtype, use_wg)));
    auto kernel = x.backend().kernels.get(kernel_name);
    std::vector<int> graph_inputs{x.id(),
                                  weight_q.id(),
                                  weight_k.id(),
                                  weight_v.id()};
    if (weight_dtype == DType::Q4_0_COL) {
        MCL_CHECK(use_wg4, "Q4_0_COL fused decode triple requires wg4 path");
        const int block = q4_0_col_block_size(weight_q);
        MCL_CHECK(q4_0_col_block_size(weight_k) == block &&
                      q4_0_col_block_size(weight_v) == block,
                  "Q4_0_COL fused decode triple expects matching block sizes");
        MCL_CHECK(weight_q.quant_scale_axis() == weight_k.quant_scale_axis() &&
                      weight_q.quant_scale_axis() == weight_v.quant_scale_axis(),
                  "Q4_0_COL fused decode triple expects matching layouts");
        MCL_CHECK(!q4_tile8 || (in % block) == 0,
                  "Q4_0_COL tile8 fused decode triple requires K divisible by block size");
        const int blocks_per_col = (in + block - 1) / block;
        auto scales_q = weight_q.quant_scales();
        auto scales_k = weight_k.quant_scales();
        auto scales_v = weight_v.quant_scales();
        graph_inputs = {x.id(),
                        weight_q.id(), scales_q.id(),
                        weight_k.id(), scales_k.id(),
                        weight_v.id(), scales_v.id()};
        kernel.set_arg(0, x.buffer());
        kernel.set_arg(1, weight_q.buffer());
        kernel.set_arg(2, scales_q.buffer());
        kernel.set_arg(3, weight_k.buffer());
        kernel.set_arg(4, scales_k.buffer());
        kernel.set_arg(5, weight_v.buffer());
        kernel.set_arg(6, scales_v.buffer());
        kernel.set_arg(7, out.q.buffer());
        kernel.set_arg(8, out.k.buffer());
        kernel.set_arg(9, out.v.buffer());
        kernel.set_arg(10, n0);
        kernel.set_arg(11, n1);
        kernel.set_arg(12, n2);
        kernel.set_arg(13, in);
        kernel.set_arg(14, blocks_per_col);
        kernel.set_arg(15, block);
        const int local = (use_q4tile8_wg64x8 || use_q4col_wg64x4) ? 64 : kLocal;
        kernel.set_arg_local(16, (use_q4tile8_wg64x8 ? 8 * local : (use_q4col_wg64x4 ? 4 * local : local)) * sizeof(float));
        const std::size_t groups = use_q4tile8_wg64x8 ? (static_cast<std::size_t>(total) + 7u) / 8u
                                                       : (static_cast<std::size_t>(total) + 3u) / 4u;
        kernel.launch1d(groups * local, local);
    } else {
        kernel.set_arg(0, x.buffer());
        kernel.set_arg(1, weight_q.buffer());
        kernel.set_arg(2, weight_k.buffer());
        kernel.set_arg(3, weight_v.buffer());
        kernel.set_arg(4, out.q.buffer());
        kernel.set_arg(5, out.k.buffer());
        kernel.set_arg(6, out.v.buffer());
        kernel.set_arg(7, n0);
        kernel.set_arg(8, n1);
        kernel.set_arg(9, n2);
        kernel.set_arg(10, in);
        if (use_wg4) {
            kernel.set_arg_local(11, kLocal * sizeof(float));
            const std::size_t groups = (static_cast<std::size_t>(total) + 3u) / 4u;
            kernel.launch1d(groups * kLocal, kLocal);
        } else if (use_wg) {
            kernel.set_arg_local(11, kLocal * sizeof(float));
            kernel.launch1d(static_cast<std::size_t>(total) * kLocal, kLocal);
        } else {
            kernel.launch1d(round_up(static_cast<std::size_t>(total), 128), 128);
        }
    }
    autograd::record_op(kernel_name, std::move(graph_inputs), {out.q.id(), out.k.id(), out.v.id()});
    return out;
}

Tensor sigmoid_gate_mul(const Tensor& x, const Tensor& gate) {
    MCL_CHECK(x.dtype() == DType::F32 && gate.dtype() == DType::F32, "sigmoid_gate_mul expects f32 tensors");
    MCL_CHECK(x.shape() == gate.shape(), "sigmoid_gate_mul shape mismatch");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get("sigmoid_gate_mul_f32");
    const int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, gate.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), static_cast<std::size_t>(256)), 256);
    return out;
}

Tensor activation_mul(const Tensor& x, const Tensor& y, const char* kernel_name) {
    MCL_CHECK(x.dtype() == DType::F32 && y.dtype() == DType::F32, "activation_mul expects f32 tensors");
    MCL_CHECK(x.shape() == y.shape(), "activation_mul shape mismatch");
    MCL_CHECK(x.backend_ptr() == y.backend_ptr(), "activation_mul requires tensors on same backend");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get(kernel_name);
    const int n = static_cast<int>(x.numel());
    k.set_arg(0, x.buffer());
    k.set_arg(1, y.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), static_cast<std::size_t>(256)), 256);
    autograd::record_op(kernel_name, {x.id(), y.id()}, {out.id()});
    return out;
}

Tensor gelu_mul_fast(const Tensor& x, const Tensor& y) {
    if (autograd::is_enabled()) return motifcl::mul(motifcl::gelu(x), y);
    return activation_mul(x, y, "gelu_mul_f32");
}

Tensor gelu_mul_packed_per_layer_fast(const Tensor& gate,
                                      const Tensor& packed,
                                      int layer,
                                      int64_t token_count) {
    MCL_CHECK(gate.dtype() == DType::F32 && packed.dtype() == DType::F32,
              "gelu_mul_packed_per_layer_fast expects f32 tensors");
    MCL_CHECK(gate.ndim() == 2 && packed.ndim() == 2,
              "gelu_mul_packed_per_layer_fast expects rank-2 tensors");
    MCL_CHECK(token_count > 0 && gate.shape()[0] == token_count,
              "gelu_mul_packed_per_layer_fast token count mismatch");
    const int64_t ple_dim = gate.shape()[1];
    MCL_CHECK(ple_dim > 0 && packed.shape()[1] == ple_dim,
              "gelu_mul_packed_per_layer_fast PLE dim mismatch");
    MCL_CHECK(packed.shape()[0] % token_count == 0,
              "gelu_mul_packed_per_layer_fast packed rows must be token_count * layer_count");
    const int64_t layer_count64 = packed.shape()[0] / token_count;
    MCL_CHECK(layer >= 0 && layer < layer_count64,
              "gelu_mul_packed_per_layer_fast layer index out of range");
    MCL_CHECK(gate.backend_ptr() == packed.backend_ptr(),
              "gelu_mul_packed_per_layer_fast requires tensors on same backend");
    auto out = Tensor::empty(gate.backend(), gate.shape(), DType::F32);
    auto k = gate.backend().kernels.get("gelu_mul_packed_per_layer_f32");
    const int n = static_cast<int>(gate.numel());
    k.set_arg(0, gate.buffer());
    k.set_arg(1, packed.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, static_cast<int>(token_count));
    k.set_arg(4, static_cast<int>(layer_count64));
    k.set_arg(5, static_cast<int>(ple_dim));
    k.set_arg(6, layer);
    k.set_arg(7, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), static_cast<std::size_t>(256)), 256);
    autograd::record_op("gelu_mul_packed_per_layer_f32", {gate.id(), packed.id()}, {out.id()});
    return out;
}

bool can_use_fused_packed_ple_gate_m1(const Tensor& h,
                                      const Tensor& packed,
                                      const Linear& gate,
                                      int layer,
                                      int64_t token_count) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_PLE_GATE_M1") || autograd::is_enabled()) return false;
    if (!h.valid() || !packed.valid() || h.dtype() != DType::F32 || packed.dtype() != DType::F32) return false;
    if (h.ndim() != 2 || packed.ndim() != 2 || h.shape()[0] != 1 || token_count != 1) return false;
    if (gate.has_bias() || gate.quantized_inference_enabled() || !gate.weight.data.valid()) return false;
    if (gate.weight.data.dtype() != DType::F32 || gate.weight.data.ndim() != 2) return false;
    const int64_t hidden = h.shape()[1];
    const int64_t ple_dim = packed.shape()[1];
    if (hidden <= 0 || ple_dim <= 0 || hidden > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        ple_dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (packed.shape()[0] <= 0 || layer < 0 || layer >= packed.shape()[0]) return false;
    if (gate.in_features() != hidden || gate.out_features() != ple_dim) return false;
    return h.backend_ptr() == packed.backend_ptr() &&
           h.backend_ptr() == gate.weight.data.backend_ptr();
}

Tensor fused_packed_ple_gate_m1(const Tensor& h,
                                const Tensor& packed,
                                const Linear& gate,
                                int layer) {
    MCL_CHECK(can_use_fused_packed_ple_gate_m1(h, packed, gate, layer, 1),
              "fused_packed_ple_gate_m1 expects compatible one-token dense PLE gate inputs");
    auto out = Tensor::empty(h.backend(), {1, gate.out_features()}, DType::F32);
    const int hidden = static_cast<int>(h.shape()[1]);
    const int ple_dim = static_cast<int>(packed.shape()[1]);
    const int layer_count = static_cast<int>(packed.shape()[0]);
    const bool use_wg64x4 = !env_enabled("MOTIFCL_DISABLE_FUSED_PLE_GATE_M1_WG64X4") &&
                            hidden >= 64 &&
                            h.backend().device_info().max_work_group_size >= 64;
    const char* kernel_name = use_wg64x4
        ? "gelu_packed_ple_gate_m1_wg64x4_f32"
        : "gelu_packed_ple_gate_m1_f32";
    auto k = h.backend().kernels.get(kernel_name);
    k.set_arg(0, h.buffer());
    k.set_arg(1, packed.buffer());
    k.set_arg(2, gate.weight.data.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, hidden);
    k.set_arg(5, ple_dim);
    k.set_arg(6, layer_count);
    k.set_arg(7, layer);
    if (use_wg64x4) {
        constexpr std::size_t kLocal = 64;
        k.set_arg_local(8, 4 * kLocal * sizeof(float));
        const std::size_t groups = (static_cast<std::size_t>(ple_dim) + 3u) / 4u;
        k.launch1d(groups * kLocal, kLocal);
    } else {
        k.launch1d(round_up(static_cast<std::size_t>(ple_dim), static_cast<std::size_t>(128)), 128);
    }
    autograd::record_op(kernel_name,
                        {h.id(), packed.id(), gate.weight.data.id()},
                        {out.id()});
    return out;
}

bool can_use_fused_packed_ple_m1(const Tensor& h,
                                 const Tensor& packed,
                                 const Linear& gate,
                                 const Linear& proj,
                                 const RMSNorm& norm,
                                 int layer,
                                 int64_t token_count) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_PLE_M1") ||
        autograd::is_enabled()) {
        return false;
    }
    if (!h.valid() || !packed.valid() || h.dtype() != DType::F32 || packed.dtype() != DType::F32) return false;
    if (h.ndim() != 2 || packed.ndim() != 2 || h.shape()[0] != 1 || token_count != 1) return false;
    if (gate.has_bias() || proj.has_bias() || gate.quantized_inference_enabled() || proj.quantized_inference_enabled()) return false;
    if (!gate.weight.data.valid() || !proj.weight.data.valid() || !norm.weight.data.valid()) return false;
    if (gate.weight.data.dtype() != DType::F32 || proj.weight.data.dtype() != DType::F32 ||
        norm.weight.data.dtype() != DType::F32) return false;
    if (gate.weight.data.ndim() != 2 || proj.weight.data.ndim() != 2 || norm.weight.data.ndim() != 1) return false;
    const int64_t hidden = h.shape()[1];
    const int64_t ple_dim = packed.shape()[1];
    if (hidden <= 0 || ple_dim <= 0 || hidden > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        ple_dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (packed.shape()[0] <= 0 || layer < 0 || layer >= packed.shape()[0]) return false;
    if (gate.in_features() != hidden || gate.out_features() != ple_dim) return false;
    if (proj.in_features() != ple_dim || proj.out_features() != hidden) return false;
    if (norm.weight.data.shape()[0] != hidden) return false;
    if (h.backend_ptr() != packed.backend_ptr() ||
        h.backend_ptr() != gate.weight.data.backend_ptr() ||
        h.backend_ptr() != proj.weight.data.backend_ptr() ||
        h.backend_ptr() != norm.weight.data.backend_ptr()) {
        return false;
    }
    constexpr std::size_t kLocal = 256;
    const auto& info = h.backend().device_info();
    if (info.max_work_group_size < kLocal) return false;
    const std::size_t local_bytes = (static_cast<std::size_t>(ple_dim) +
                                     static_cast<std::size_t>(hidden) +
                                     kLocal) * sizeof(float);
    return local_bytes <= info.local_mem_size;
}

Tensor fused_packed_ple_m1(const Tensor& h,
                           const Tensor& packed,
                           const Linear& gate,
                           const Linear& proj,
                           const RMSNorm& norm,
                           int layer) {
    MCL_CHECK(can_use_fused_packed_ple_m1(h, packed, gate, proj, norm, layer, 1),
              "fused_packed_ple_m1 expects compatible one-token dense PLE inputs");
    auto out = Tensor::empty(h.backend(), h.shape(), DType::F32);
    auto k = h.backend().kernels.get("fused_packed_ple_m1_f32");
    constexpr std::size_t kLocal = 256;
    const int hidden = static_cast<int>(h.shape()[1]);
    const int ple_dim = static_cast<int>(packed.shape()[1]);
    const int layer_count = static_cast<int>(packed.shape()[0]);
    const std::size_t local_bytes = (static_cast<std::size_t>(ple_dim) +
                                     static_cast<std::size_t>(hidden) +
                                     kLocal) * sizeof(float);
    k.set_arg(0, h.buffer());
    k.set_arg(1, packed.buffer());
    k.set_arg(2, gate.weight.data.buffer());
    k.set_arg(3, proj.weight.data.buffer());
    k.set_arg(4, norm.weight.data.buffer());
    k.set_arg(5, out.buffer());
    k.set_arg(6, hidden);
    k.set_arg(7, ple_dim);
    k.set_arg(8, layer_count);
    k.set_arg(9, layer);
    k.set_arg(10, norm.eps);
    k.set_arg_local(11, local_bytes);
    k.launch1d(kLocal, kLocal);
    autograd::record_op("fused_packed_ple_m1_f32",
                        {h.id(), packed.id(), gate.weight.data.id(), proj.weight.data.id(), norm.weight.data.id()},
                        {out.id()});
    return out;
}

Tensor silu_mul_fast(const Tensor& x, const Tensor& y) {
    if (autograd::is_enabled()) return motifcl::mul(motifcl::silu(x), y);
    return activation_mul(x, y, "silu_mul_f32");
}

Tensor moe_swiglu_forward(const Tensor& x,
                          const Tensor& router_weight,
                          const Tensor& gate_weight,
                          const Tensor& up_weight,
                          const Tensor& down_weight,
                          int experts_per_token) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "moe_swiglu_forward expects f32 [tokens, in] input");
    MCL_CHECK(router_weight.dtype() == DType::F32 && router_weight.ndim() == 2, "moe_swiglu_forward expects router [in, experts]");
    MCL_CHECK(gate_weight.dtype() == DType::F32 && up_weight.dtype() == DType::F32 &&
                  down_weight.dtype() == DType::F32,
              "moe_swiglu_forward expert weights must be f32");
    MCL_CHECK(gate_weight.ndim() == 3 && up_weight.ndim() == 3 && down_weight.ndim() == 3,
              "moe_swiglu_forward expert weights must be rank-3");
    const int64_t rows = x.shape()[0];
    const int64_t in = x.shape()[1];
    const int64_t experts = router_weight.shape()[1];
    const int64_t hidden = gate_weight.shape()[2];
    MCL_CHECK(router_weight.shape()[0] == in, "moe_swiglu_forward router input dimension mismatch");
    MCL_CHECK(gate_weight.shape()[0] == experts && up_weight.shape()[0] == experts &&
                  down_weight.shape()[0] == experts,
              "moe_swiglu_forward expert count mismatch");
    MCL_CHECK(gate_weight.shape()[1] == in && up_weight.shape()[1] == in,
              "moe_swiglu_forward gate/up input dimension mismatch");
    MCL_CHECK(up_weight.shape()[2] == hidden && down_weight.shape()[1] == hidden && down_weight.shape()[2] == in,
              "moe_swiglu_forward expert hidden/output dimension mismatch");
    MCL_CHECK(experts_per_token > 0 && experts_per_token <= experts && experts_per_token <= 8,
              "moe_swiglu_forward supports top_k in [1, min(experts, 8)]");
    auto out = Tensor::empty(x.backend(), {rows, in}, DType::F32);
    auto k = x.backend().kernels.get("moe_swiglu_forward_f32");
    k.set_arg(0, x.buffer());
    k.set_arg(1, router_weight.buffer());
    k.set_arg(2, gate_weight.buffer());
    k.set_arg(3, up_weight.buffer());
    k.set_arg(4, down_weight.buffer());
    k.set_arg(5, out.buffer());
    k.set_arg(6, static_cast<int>(rows));
    k.set_arg(7, static_cast<int>(in));
    k.set_arg(8, static_cast<int>(hidden));
    k.set_arg(9, static_cast<int>(experts));
    k.set_arg(10, experts_per_token);
    k.launch2d(round_up(static_cast<std::size_t>(in), 16), round_up(static_cast<std::size_t>(rows), 16), 16, 16);
    return out;
}

Tensor gated_delta_recurrent(const Tensor& q,
                             const Tensor& k_new,
                             const Tensor& v,
                             const Tensor& gate,
                             Tensor& state,
                             int64_t batch_size,
                             int64_t seq_len,
                             int n_head,
                             int head_dim,
                             float decay) {
    MCL_CHECK(q.dtype() == DType::F32 && k_new.dtype() == DType::F32 &&
                  v.dtype() == DType::F32 && gate.dtype() == DType::F32,
              "gated_delta_recurrent expects f32 projections");
    MCL_CHECK(q.shape() == k_new.shape() && q.shape() == v.shape() && q.shape() == gate.shape(),
              "gated_delta_recurrent projection shape mismatch");
    MCL_CHECK(q.ndim() == 2 && q.shape()[0] == batch_size * seq_len &&
                  q.shape()[1] == static_cast<int64_t>(n_head) * head_dim,
              "gated_delta_recurrent expects flattened [B*T, n_head*head_dim] projections");
    MCL_CHECK(state.valid() && state.dtype() == DType::F32 && state.ndim() == 4,
              "gated_delta_recurrent expects f32 state [B,H,D,S]");
    MCL_CHECK(state.shape()[0] == batch_size && state.shape()[1] == n_head &&
                  state.shape()[2] == head_dim && state.shape()[3] == head_dim,
              "gated_delta_recurrent currently requires state_dim == head_dim");
    auto out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("gated_delta_recurrent_f32");
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k_new.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, gate.buffer());
    kernel.set_arg(4, state.buffer());
    kernel.set_arg(5, out.buffer());
    kernel.set_arg(6, static_cast<int>(batch_size));
    kernel.set_arg(7, static_cast<int>(seq_len));
    kernel.set_arg(8, n_head);
    kernel.set_arg(9, head_dim);
    kernel.set_arg(10, decay);
    kernel.launch2d(round_up(static_cast<std::size_t>(n_head * head_dim), 16),
                    round_up(static_cast<std::size_t>(batch_size), 16), 16, 16);
    return out;
}
}

KVCache::KVCache(Backend& backend, int64_t batch, int64_t max_seq, int n_kv, int head_dim_value)
    : k(Tensor::empty(backend, {batch * max_seq, n_kv * head_dim_value}, DType::F32)),
      v(Tensor::empty(backend, {batch * max_seq, n_kv * head_dim_value}, DType::F32)),
      batch_size(batch),
      max_seq_len(max_seq),
      n_kv_head(n_kv),
      head_dim(head_dim_value) {
    MCL_CHECK(batch > 0 && max_seq > 0 && n_kv > 0 && head_dim_value > 0, "KVCache invalid dimensions");
}

PagedKVCache::PagedKVCache(Backend& backend, int64_t batch, int64_t max_seq, int64_t page_size_value,
                           int n_kv, int head_dim_value)
    : batch_size(batch),
      max_seq_len(max_seq),
      page_size(page_size_value),
      page_count((max_seq + page_size_value - 1) / page_size_value),
      n_kv_head(n_kv),
      head_dim(head_dim_value) {
    MCL_CHECK(batch > 0 && max_seq > 0 && page_size_value > 0 && n_kv > 0 && head_dim_value > 0,
              "PagedKVCache invalid dimensions");
    const int64_t channels = static_cast<int64_t>(n_kv) * head_dim_value;
    k_pages = Tensor::empty(backend, {batch * page_count * page_size, channels}, DType::F32);
    v_pages = Tensor::empty(backend, {batch * page_count * page_size, channels}, DType::F32);
    std::vector<std::int32_t> table(static_cast<std::size_t>(batch * page_count));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t p = 0; p < page_count; ++p) {
            table[static_cast<std::size_t>(b * page_count + p)] =
                static_cast<std::int32_t>(b * page_count + p);
        }
    }
    page_table = Tensor::from_cpu(backend, {batch, page_count}, DType::I32, table.data());
}

DeltaStateCache::DeltaStateCache(Backend& backend, int64_t batch, int heads, int head_dim_value, int state_dim_value)
    : state(Tensor::zeros(backend,
                          {batch, static_cast<int64_t>(heads), head_dim_value, state_dim_value},
                          DType::F32)),
      batch_size(batch),
      num_heads(heads),
      head_dim(head_dim_value),
      state_dim(state_dim_value) {
    MCL_CHECK(batch > 0 && heads > 0 && head_dim_value > 0 && state_dim_value > 0,
              "DeltaStateCache invalid dimensions");
}

void DeltaStateCache::zero() {
    MCL_CHECK(state.valid(), "DeltaStateCache is not initialized");
    state = Tensor::zeros(state.backend(), state.shape(), DType::F32);
}

ModernMLP::ModernMLP(Backend& backend, int n_embd, int hidden, bool swiglu_enabled,
                     bool use_bias, float dropout, bool skip_weight_init)
    : gate_up_proj(backend, n_embd, swiglu_enabled ? hidden * 2 : hidden, use_bias, skip_weight_init),
      down_proj(backend, hidden, n_embd, use_bias, skip_weight_init),
      use_swiglu(swiglu_enabled),
      dropout_p(dropout),
      gate_proj_(backend, n_embd, hidden, use_bias, skip_weight_init),
      up_proj_(backend, n_embd, hidden, use_bias, skip_weight_init),
      use_split_projections_(false) {}

Tensor ModernMLP::forward(const Tensor& x) {
    Tensor hidden;
    if (use_split_projections_ ||
        gate_proj_.quantized_inference_enabled() ||
        up_proj_.quantized_inference_enabled()) {
        Tensor gate;
        Tensor up;
        if (use_swiglu && dropout_p == 0.0f &&
            can_use_fused_decode_swiglu_product_pair(x, gate_proj_, up_proj_)) {
            hidden = fused_decode_swiglu_product_pair(x, gate_proj_, up_proj_);
            auto out = down_proj.forward(hidden);
            return out;
        }
        if (can_use_fused_decode_projection_pair(x, gate_proj_, up_proj_)) {
            auto fused = fused_decode_projection_pair(x, gate_proj_, up_proj_);
            gate = std::move(fused.first);
            up = std::move(fused.second);
        } else {
            gate = gate_proj_.forward(x);
            up = up_proj_.forward(x);
        }
        if (use_swiglu && dropout_p == 0.0f && can_use_fused_silu_down_q4_0_col(gate, up, down_proj)) {
            return fused_silu_down_q4_0_col(gate, up, down_proj);
        }
        hidden = use_swiglu ? silu_mul_fast(gate, up) : gelu_mul_fast(gate, up);
    } else {
        hidden = gate_up_proj.forward(x);
        hidden = use_swiglu ? motifcl::swiglu(hidden) : motifcl::gelu(hidden);
    }
    auto out = down_proj.forward(hidden);
    return dropout_p > 0.0f ? motifcl::dropout(out, dropout_p, true) : out;
}


Tensor mlp_forward_rmsnorm_decode(const Tensor& h, RMSNorm& norm, ModernMLP& mlp) {
    if (mlp.split_projections_enabled() &&
        mlp.use_swiglu &&
        mlp.dropout_p == 0.0f &&
        (mlp.gate_proj().quantized_inference_enabled() || mlp.up_proj().quantized_inference_enabled()) &&
        !env_enabled("MOTIFCL_DISABLE_FUSED_SWIGLU_PRODUCT_PAIR")) {
        auto x = norm.forward(h);
        if (can_use_fused_decode_swiglu_product_pair(x, mlp.gate_proj(), mlp.up_proj())) {
            auto hidden = fused_decode_swiglu_product_pair(x, mlp.gate_proj(), mlp.up_proj());
            return mlp.down_proj.forward(hidden);
        }
        if (!env_enabled("MOTIFCL_ENABLE_FUSED_RMSNORM_DECODE_PAIR")) return mlp.forward(x);
    }
    if (mlp.split_projections_enabled() &&
        (mlp.gate_proj().quantized_inference_enabled() || mlp.up_proj().quantized_inference_enabled()) &&
        can_use_fused_decode_projection_pair_rmsnorm(h, norm, mlp.gate_proj(), mlp.up_proj())) {
        auto fused = fused_decode_projection_pair_rmsnorm(h, norm, mlp.gate_proj(), mlp.up_proj());
        Tensor gate = std::move(fused.first);
        Tensor up = std::move(fused.second);
        Tensor out;
        if (mlp.use_swiglu && mlp.dropout_p == 0.0f && can_use_fused_silu_down_q4_0_col(gate, up, mlp.down_proj)) {
            out = fused_silu_down_q4_0_col(gate, up, mlp.down_proj);
        } else {
            Tensor hidden = mlp.use_swiglu ? silu_mul_fast(gate, up) : gelu_mul_fast(gate, up);
            out = mlp.down_proj.forward(hidden);
        }
        return mlp.dropout_p > 0.0f ? motifcl::dropout(out, mlp.dropout_p, true) : out;
    }
    return mlp.forward(norm.forward(h));
}

std::vector<Parameter*> ModernMLP::parameters() {
    std::vector<Parameter*> result;
    if (use_split_projections_) {
        auto pg = gate_proj_.parameters();
        auto pu = up_proj_.parameters();
        result.insert(result.end(), pg.begin(), pg.end());
        result.insert(result.end(), pu.begin(), pu.end());
    } else {
        auto p = gate_up_proj.parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto pd = down_proj.parameters();
    result.insert(result.end(), pd.begin(), pd.end());
    return result;
}

void ModernMLP::enable_quantized_inference(DType qdtype) {
    if (use_split_projections_) {
        gate_proj_.enable_quantized_inference(qdtype);
        up_proj_.enable_quantized_inference(qdtype);
    } else {
        gate_up_proj.enable_quantized_inference(qdtype);
    }
    down_proj.enable_quantized_inference(qdtype);
}

void ModernMLP::disable_quantized_inference() {
    gate_up_proj.disable_quantized_inference();
    gate_proj_.disable_quantized_inference();
    up_proj_.disable_quantized_inference();
    down_proj.disable_quantized_inference();
}

bool ModernMLP::quantized_inference_enabled() const {
    const bool input_proj =
        use_split_projections_
            ? (gate_proj_.quantized_inference_enabled() && up_proj_.quantized_inference_enabled())
            : gate_up_proj.quantized_inference_enabled();
    return input_proj && down_proj.quantized_inference_enabled();
}

DType ModernMLP::quantized_weight_dtype() const {
    if (!quantized_inference_enabled()) return DType::F32;
    return use_split_projections_ ? gate_proj_.quantized_weight_dtype() : gate_up_proj.quantized_weight_dtype();
}

MoEFFN::MoEFFN(Backend& backend, int n_embd, int hidden, int experts, int top_k)
    : router_weight(Tensor::randn(backend, {n_embd, experts}, 1.0f / std::sqrt(static_cast<float>(n_embd)))),
      expert_gate_weight(Tensor::randn(backend, {experts, n_embd, hidden}, 1.0f / std::sqrt(static_cast<float>(n_embd)))),
      expert_up_weight(Tensor::randn(backend, {experts, n_embd, hidden}, 1.0f / std::sqrt(static_cast<float>(n_embd)))),
      expert_down_weight(Tensor::randn(backend, {experts, hidden, n_embd}, 1.0f / std::sqrt(static_cast<float>(hidden)))),
      num_experts(experts),
      experts_per_token(top_k),
      in_features(n_embd),
      hidden_features(hidden) {
    MCL_CHECK(n_embd > 0 && hidden > 0 && experts > 0 && top_k > 0 && top_k <= experts,
              "MoEFFN invalid dimensions");
}

Tensor MoEFFN::forward(const Tensor& x) {
    return moe_swiglu_forward(x, router_weight.data, expert_gate_weight.data, expert_up_weight.data,
                              expert_down_weight.data, experts_per_token);
}

std::vector<Parameter*> MoEFFN::parameters() {
    return {&router_weight, &expert_gate_weight, &expert_up_weight, &expert_down_weight};
}

GatedDeltaNetLayer::GatedDeltaNetLayer(Backend& backend, const TransformerConfig& raw_config, int state_dim)
    : q_proj(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, false),
      k_proj(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, false),
      v_proj(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, false),
      gate_proj(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, false),
      o_proj(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, false) {
    const auto cfg = normalize_config(raw_config);
    n_embd_ = cfg.n_embd;
    n_head_ = cfg.n_head;
    head_dim_ = cfg.n_embd / cfg.n_head;
    state_dim_ = state_dim > 0 ? state_dim : head_dim_;
    MCL_CHECK(state_dim_ == head_dim_, "GatedDeltaNetLayer currently requires state_dim == head_dim");
}

Tensor GatedDeltaNetLayer::forward(const Tensor& x) {
    MCL_CHECK(x.ndim() == 2 && x.shape()[1] == n_embd_, "GatedDeltaNetLayer expects [T,n_embd] input");
    DeltaStateCache state(x.backend(), 1, n_head_, head_dim_, state_dim_);
    return forward_with_state(x, state, 1, x.shape()[0]);
}

Tensor GatedDeltaNetLayer::forward_with_state(const Tensor& x, DeltaStateCache& state, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(x.ndim() == 2 && x.shape()[0] == batch_size * seq_len && x.shape()[1] == n_embd_,
              "GatedDeltaNetLayer flattened input shape mismatch");
    MCL_CHECK(state.batch_size == batch_size && state.num_heads == n_head_ &&
                  state.head_dim == head_dim_ && state.state_dim == state_dim_,
              "GatedDeltaNetLayer state-cache shape mismatch");
    auto q = q_proj.forward(x);
    auto k = k_proj.forward(x);
    auto v = v_proj.forward(x);
    auto g = gate_proj.forward(x);
    auto recurrent = gated_delta_recurrent(q, k, v, g, state.state, batch_size, seq_len, n_head_, head_dim_, decay_);
    return o_proj.forward(recurrent);
}

std::vector<Parameter*> GatedDeltaNetLayer::parameters() {
    std::vector<Parameter*> result;
    for (auto* layer : {&q_proj, &k_proj, &v_proj, &gate_proj, &o_proj}) {
        auto p = layer->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

ModernSelfAttention::ModernSelfAttention(Backend& backend, const TransformerConfig& raw_config)
    : qkv_proj_(backend,
                normalize_config(raw_config).n_embd,
                normalize_config(raw_config).n_head * normalize_config(raw_config).head_dim +
                    2 * normalize_config(raw_config).n_kv_head * normalize_config(raw_config).head_dim,
                normalize_config(raw_config).use_qkv_bias,
                normalize_config(raw_config).skip_weight_init),
      q_proj_(backend,
              normalize_config(raw_config).n_embd,
              normalize_config(raw_config).n_head * normalize_config(raw_config).head_dim,
              normalize_config(raw_config).use_qkv_bias,
              normalize_config(raw_config).skip_weight_init),
      k_proj_(backend,
              normalize_config(raw_config).n_embd,
              normalize_config(raw_config).n_kv_head * normalize_config(raw_config).head_dim,
              normalize_config(raw_config).use_qkv_bias,
              normalize_config(raw_config).skip_weight_init),
      v_proj_(backend,
              normalize_config(raw_config).n_embd,
              normalize_config(raw_config).n_kv_head * normalize_config(raw_config).head_dim,
              normalize_config(raw_config).use_qkv_bias,
              normalize_config(raw_config).skip_weight_init),
      o_proj_(backend,
              normalize_config(raw_config).n_head * normalize_config(raw_config).head_dim,
              normalize_config(raw_config).n_embd,
              normalize_config(raw_config).use_qkv_bias,
              normalize_config(raw_config).skip_weight_init),
      q_norm_(backend, normalize_config(raw_config).head_dim, normalize_config(raw_config).rms_norm_eps),
      k_norm_(backend, normalize_config(raw_config).head_dim, normalize_config(raw_config).rms_norm_eps) {
    const auto cfg = normalize_config(raw_config);
    n_embd_ = cfg.n_embd;
    n_head_ = cfg.n_head;
    n_kv_head_ = cfg.n_kv_head;
    head_dim_ = cfg.head_dim;
    q_dim_ = cfg.n_head * head_dim_;
    kv_dim_ = cfg.n_kv_head * head_dim_;
    use_rope_ = cfg.use_rope;
    rope_theta_ = cfg.rope_theta;
    rotary_dim_ = cfg.rotary_dim;
    dropout_p_ = cfg.dropout;
    attention_window_ = cfg.sliding_window;
    use_split_projections_ = cfg.split_qkv_projections;
    use_qk_norm_ = cfg.use_qk_norm;
}

QKV ModernSelfAttention::project_qkv(const Tensor& x) {
    if (use_split_projections_ ||
        q_proj_.quantized_inference_enabled() ||
        k_proj_.quantized_inference_enabled() ||
        v_proj_.quantized_inference_enabled()) {
        if (can_use_fused_decode_projection_triple(x, q_proj_, k_proj_, v_proj_)) {
            return fused_decode_projection_triple(x, q_proj_, k_proj_, v_proj_);
        }
        return {q_proj_.forward(x), k_proj_.forward(x), v_proj_.forward(x)};
    }
    auto packed = qkv_proj_.forward(x);
    return motifcl::qkv_split(packed, q_dim_, kv_dim_);
}

void apply_qk_norm_if_enabled(ModernSelfAttention& self, Tensor& q, Tensor& k) {
    if (!self.qk_norm_enabled()) return;
    const auto rows = q.shape()[0];
    const auto q_shape = q.shape();
    const auto k_shape = k.shape();
    q = self.q_norm().forward(q.view({rows * self.n_head(), self.head_dim()})).view(q_shape);
    k = self.k_norm().forward(k.view({rows * self.n_kv_head(), self.head_dim()})).view(k_shape);
}

bool can_use_fused_qk_norm_rope_decode(const Tensor& q,
                                       const Tensor& k,
                                       const RMSNorm& q_norm,
                                       const RMSNorm& k_norm,
                                       int n_head,
                                       int n_kv_head,
                                       int head_dim,
                                       int64_t batch_size,
                                       int64_t seq_len,
                                       bool use_rope,
                                       int rotary_dim) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_DECODE") || autograd::is_enabled()) return false;
    if (!use_rope || !q.valid() || !k.valid() || !q_norm.weight.data.valid() || !k_norm.weight.data.valid()) return false;
    if (q.dtype() != DType::F32 || k.dtype() != DType::F32 ||
        q_norm.weight.data.dtype() != DType::F32 || k_norm.weight.data.dtype() != DType::F32) {
        return false;
    }
    if (q.ndim() != 2 || k.ndim() != 2 ||
        q_norm.weight.data.ndim() != 1 || k_norm.weight.data.ndim() != 1) {
        return false;
    }
    if (batch_size <= 0 || seq_len != 1 || n_head <= 0 || n_kv_head <= 0 || head_dim <= 0) return false;
    if (q.shape()[0] != batch_size || k.shape()[0] != batch_size) return false;
    if (q.shape()[1] != static_cast<int64_t>(n_head) * head_dim ||
        k.shape()[1] != static_cast<int64_t>(n_kv_head) * head_dim) {
        return false;
    }
    if (q_norm.weight.data.shape()[0] != head_dim || k_norm.weight.data.shape()[0] != head_dim) return false;
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = std::min(rd, head_dim);
    if (rd & 1) return false;
    if (q.backend_ptr() != k.backend_ptr() ||
        q.backend_ptr() != q_norm.weight.data.backend_ptr() ||
        q.backend_ptr() != k_norm.weight.data.backend_ptr()) {
        return false;
    }
    constexpr std::size_t kLocal = 256;
    return q.backend().device_info().max_work_group_size >= kLocal;
}

std::pair<Tensor, Tensor> fused_qk_norm_rope_decode(const Tensor& q,
                                                    const Tensor& k,
                                                    const RMSNorm& q_norm,
                                                    const RMSNorm& k_norm,
                                                    int n_head,
                                                    int n_kv_head,
                                                    int head_dim,
                                                    int64_t batch_size,
                                                    int64_t token_offset,
                                                    float theta,
                                                    int rotary_dim) {
    MCL_CHECK(can_use_fused_qk_norm_rope_decode(q, k, q_norm, k_norm, n_head, n_kv_head,
                                                head_dim, batch_size, 1, true, rotary_dim),
              "fused_qk_norm_rope_decode expects compatible one-token q/k projections");
    auto q_out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto k_out = Tensor::empty(q.backend(), k.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("qk_norm_rope_decode_f32");
    constexpr std::size_t kLocal = 256;
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, q_norm.weight.data.buffer());
    kernel.set_arg(3, k_norm.weight.data.buffer());
    kernel.set_arg(4, q_out.buffer());
    kernel.set_arg(5, k_out.buffer());
    kernel.set_arg(6, static_cast<int>(batch_size));
    kernel.set_arg(7, n_head);
    kernel.set_arg(8, n_kv_head);
    kernel.set_arg(9, head_dim);
    kernel.set_arg(10, rd);
    kernel.set_arg(11, static_cast<int>(token_offset));
    kernel.set_arg(12, theta);
    kernel.set_arg(13, q_norm.eps);
    kernel.set_arg(14, k_norm.eps);
    kernel.set_arg_local(15, kLocal * sizeof(float));
    const std::size_t groups = static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(n_head + n_kv_head);
    kernel.launch1d(groups * kLocal, kLocal);
    autograd::record_op("qk_norm_rope_decode_f32",
                        {q.id(), k.id(), q_norm.weight.data.id(), k_norm.weight.data.id()},
                        {q_out.id(), k_out.id()});
    return {q_out, k_out};
}


bool can_use_fused_qk_norm_rope_cache_append_decode(const Tensor& q,
                                                    const Tensor& k,
                                                    const Tensor& v,
                                                    const RMSNorm& q_norm,
                                                    const RMSNorm& k_norm,
                                                    const Tensor& cache_k,
                                                    const Tensor& cache_v,
                                                    int n_head,
                                                    int n_kv_head,
                                                    int head_dim,
                                                    int64_t batch_size,
                                                    int64_t seq_len,
                                                    int64_t max_seq_len,
                                                    int64_t token_offset,
                                                    bool use_rope,
                                                    int rotary_dim) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_CACHE_APPEND_DECODE") || autograd::is_enabled()) return false;
    if (!can_use_fused_qk_norm_rope_decode(q, k, q_norm, k_norm, n_head, n_kv_head,
                                           head_dim, batch_size, seq_len, use_rope, rotary_dim)) {
        return false;
    }
    if (!v.valid() || !cache_k.valid() || !cache_v.valid()) return false;
    if (v.dtype() != DType::F32 || cache_k.dtype() != DType::F32 || cache_v.dtype() != DType::F32) return false;
    if (v.ndim() != 2 || cache_k.ndim() != 2 || cache_v.ndim() != 2) return false;
    if (batch_size <= 0 || seq_len != 1 || max_seq_len <= 0 || token_offset < 0 || token_offset >= max_seq_len) return false;
    if (batch_size > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        max_seq_len > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        token_offset > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const int64_t kv_channels = static_cast<int64_t>(n_kv_head) * head_dim;
    if (v.shape()[0] != batch_size || v.shape()[1] != kv_channels) return false;
    if (cache_k.shape()[0] != batch_size * max_seq_len || cache_k.shape()[1] != kv_channels) return false;
    if (cache_v.shape() != cache_k.shape()) return false;
    return q.backend_ptr() == v.backend_ptr() &&
           q.backend_ptr() == cache_k.backend_ptr() &&
           q.backend_ptr() == cache_v.backend_ptr();
}

Tensor fused_qk_norm_rope_cache_append_decode(const Tensor& q,
                                              const Tensor& k,
                                              const Tensor& v,
                                              const RMSNorm& q_norm,
                                              const RMSNorm& k_norm,
                                              Tensor& cache_k,
                                              Tensor& cache_v,
                                              int n_head,
                                              int n_kv_head,
                                              int head_dim,
                                              int64_t batch_size,
                                              int64_t max_seq_len,
                                              int64_t token_offset,
                                              float theta,
                                              int rotary_dim) {
    MCL_CHECK(can_use_fused_qk_norm_rope_cache_append_decode(q, k, v, q_norm, k_norm, cache_k, cache_v,
                                                            n_head, n_kv_head, head_dim, batch_size, 1,
                                                            max_seq_len, token_offset, true, rotary_dim),
              "fused_qk_norm_rope_cache_append_decode expects compatible one-token q/k/v projections and KV cache");
    auto q_out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("qk_norm_rope_cache_append_decode_f32");
    constexpr std::size_t kLocal = 256;
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, q_norm.weight.data.buffer());
    kernel.set_arg(4, k_norm.weight.data.buffer());
    kernel.set_arg(5, q_out.buffer());
    kernel.set_arg(6, cache_k.buffer());
    kernel.set_arg(7, cache_v.buffer());
    kernel.set_arg(8, static_cast<int>(batch_size));
    kernel.set_arg(9, n_head);
    kernel.set_arg(10, n_kv_head);
    kernel.set_arg(11, head_dim);
    kernel.set_arg(12, rd);
    kernel.set_arg(13, static_cast<int>(token_offset));
    kernel.set_arg(14, static_cast<int>(max_seq_len));
    kernel.set_arg(15, theta);
    kernel.set_arg(16, q_norm.eps);
    kernel.set_arg(17, k_norm.eps);
    kernel.set_arg_local(18, kLocal * sizeof(float));
    const std::size_t groups = static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(n_head + 2 * n_kv_head);
    kernel.launch1d(groups * kLocal, kLocal);
    autograd::record_op("qk_norm_rope_cache_append_decode_f32",
                        {q.id(), k.id(), v.id(), q_norm.weight.data.id(), k_norm.weight.data.id()},
                        {q_out.id(), cache_k.id(), cache_v.id()});
    return q_out;
}

bool can_use_fused_rope_cache_append_decode(const Tensor& q,
                                            const Tensor& k,
                                            const Tensor& v,
                                            const Tensor& cache_k,
                                            const Tensor& cache_v,
                                            int n_head,
                                            int n_kv_head,
                                            int head_dim,
                                            int64_t batch_size,
                                            int64_t seq_len,
                                            int64_t max_seq_len,
                                            int64_t token_offset,
                                            bool use_rope,
                                            int rotary_dim) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_ROPE_CACHE_APPEND_DECODE") || autograd::is_enabled()) return false;
    if (!use_rope || !q.valid() || !k.valid() || !v.valid() || !cache_k.valid() || !cache_v.valid()) return false;
    if (q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 ||
        cache_k.dtype() != DType::F32 || cache_v.dtype() != DType::F32) {
        return false;
    }
    if (q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 || cache_k.ndim() != 2 || cache_v.ndim() != 2) return false;
    if (batch_size <= 0 || seq_len != 1 || n_head <= 0 || n_kv_head <= 0 ||
        head_dim <= 0 || max_seq_len <= 0 || token_offset < 0 || token_offset >= max_seq_len) {
        return false;
    }
    if (batch_size > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        max_seq_len > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        token_offset > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const int64_t q_channels = static_cast<int64_t>(n_head) * head_dim;
    const int64_t kv_channels = static_cast<int64_t>(n_kv_head) * head_dim;
    if (q.shape()[0] != batch_size || q.shape()[1] != q_channels) return false;
    if (k.shape()[0] != batch_size || k.shape()[1] != kv_channels) return false;
    if (v.shape() != k.shape()) return false;
    if (cache_k.shape()[0] != batch_size * max_seq_len || cache_k.shape()[1] != kv_channels) return false;
    if (cache_v.shape() != cache_k.shape()) return false;
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = std::min(rd, head_dim);
    if (rd & 1) return false;
    const int64_t total = batch_size * (q_channels + 2 * kv_channels);
    if (total <= 0 || total > static_cast<int64_t>(std::numeric_limits<int>::max())) return false;
    return q.backend_ptr() == k.backend_ptr() &&
           q.backend_ptr() == v.backend_ptr() &&
           q.backend_ptr() == cache_k.backend_ptr() &&
           q.backend_ptr() == cache_v.backend_ptr();
}

Tensor fused_rope_cache_append_decode(const Tensor& q,
                                      const Tensor& k,
                                      const Tensor& v,
                                      Tensor& cache_k,
                                      Tensor& cache_v,
                                      int n_head,
                                      int n_kv_head,
                                      int head_dim,
                                      int64_t batch_size,
                                      int64_t max_seq_len,
                                      int64_t token_offset,
                                      float theta,
                                      int rotary_dim) {
    MCL_CHECK(can_use_fused_rope_cache_append_decode(q, k, v, cache_k, cache_v, n_head, n_kv_head, head_dim,
                                                     batch_size, 1, max_seq_len, token_offset, true, rotary_dim),
              "fused_rope_cache_append_decode expects compatible one-token q/k/v projections and KV cache");
    auto q_out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("rope_cache_append_decode_f32");
    const int q_channels = n_head * head_dim;
    const int kv_channels = n_kv_head * head_dim;
    const int total = static_cast<int>(batch_size) * (q_channels + 2 * kv_channels);
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, q_out.buffer());
    kernel.set_arg(4, cache_k.buffer());
    kernel.set_arg(5, cache_v.buffer());
    kernel.set_arg(6, static_cast<int>(batch_size));
    kernel.set_arg(7, n_head);
    kernel.set_arg(8, n_kv_head);
    kernel.set_arg(9, head_dim);
    kernel.set_arg(10, rd);
    kernel.set_arg(11, static_cast<int>(token_offset));
    kernel.set_arg(12, static_cast<int>(max_seq_len));
    kernel.set_arg(13, theta);
    kernel.set_arg(14, total);
    constexpr std::size_t kLocal = 128;
    kernel.launch1d(round_up(static_cast<std::size_t>(total), kLocal), kLocal);
    autograd::record_op("rope_cache_append_decode_f32",
                        {q.id(), k.id(), v.id()},
                        {q_out.id(), cache_k.id(), cache_v.id()});
    return q_out;
}

bool can_use_fused_rmsnorm_residual_add(const Tensor& residual, const Tensor& x, const RMSNorm& norm) {
    if (env_enabled("MOTIFCL_DISABLE_FUSED_RMSNORM_RESIDUAL_ADD") || autograd::is_enabled()) return false;
    if (!residual.valid() || !x.valid() || !norm.weight.data.valid()) return false;
    if (residual.dtype() != DType::F32 || x.dtype() != DType::F32 || norm.weight.data.dtype() != DType::F32) return false;
    if (residual.ndim() != 2 || x.ndim() != 2 || norm.weight.data.ndim() != 1) return false;
    if (residual.shape() != x.shape() || x.shape()[1] != norm.weight.data.shape()[0]) return false;
    if (residual.backend_ptr() != x.backend_ptr() || residual.backend_ptr() != norm.weight.data.backend_ptr()) return false;
    constexpr std::size_t kLocal = 256;
    return residual.backend().device_info().max_work_group_size >= kLocal;
}

bool can_use_fused_rmsnorm_residual_add_scale(const Tensor& residual,
                                              const Tensor& x,
                                              const RMSNorm& norm,
                                              const Tensor& scale) {
    if (!can_use_fused_rmsnorm_residual_add(residual, x, norm)) return false;
    if (!scale.valid() || scale.dtype() != DType::F32 || scale.ndim() != 1) return false;
    if (scale.shape()[0] != 1 && scale.shape()[0] != x.shape()[1]) return false;
    return scale.backend_ptr() == residual.backend_ptr();
}

Tensor fused_rmsnorm_residual_add(const Tensor& residual, const Tensor& x, const RMSNorm& norm) {
    MCL_CHECK(can_use_fused_rmsnorm_residual_add(residual, x, norm),
              "fused_rmsnorm_residual_add expects compatible f32 [rows, cols] tensors");
    auto out = Tensor::empty(residual.backend(), residual.shape(), DType::F32);
    auto kernel = residual.backend().kernels.get("rmsnorm_residual_add_rowwise_wg_f32");
    constexpr std::size_t kLocal = 256;
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);
    kernel.set_arg(0, residual.buffer());
    kernel.set_arg(1, x.buffer());
    kernel.set_arg(2, norm.weight.data.buffer());
    kernel.set_arg(3, out.buffer());
    kernel.set_arg(4, rows);
    kernel.set_arg(5, cols);
    kernel.set_arg(6, norm.eps);
    kernel.launch2d(kLocal, static_cast<std::size_t>(rows), kLocal, 1);
    autograd::record_op("rmsnorm_residual_add_rowwise_wg_f32",
                        {residual.id(), x.id(), norm.weight.data.id()},
                        {out.id()});
    return out;
}

Tensor fused_rmsnorm_residual_add_scale(const Tensor& residual,
                                        const Tensor& x,
                                        const RMSNorm& norm,
                                        const Tensor& scale) {
    MCL_CHECK(can_use_fused_rmsnorm_residual_add_scale(residual, x, norm, scale),
              "fused_rmsnorm_residual_add_scale expects compatible f32 tensors");
    auto out = Tensor::empty(residual.backend(), residual.shape(), DType::F32);
    auto kernel = residual.backend().kernels.get("rmsnorm_residual_add_scale_rowwise_wg_f32");
    constexpr std::size_t kLocal = 256;
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);
    const int scale_size = static_cast<int>(scale.shape()[0]);
    kernel.set_arg(0, residual.buffer());
    kernel.set_arg(1, x.buffer());
    kernel.set_arg(2, norm.weight.data.buffer());
    kernel.set_arg(3, scale.buffer());
    kernel.set_arg(4, out.buffer());
    kernel.set_arg(5, rows);
    kernel.set_arg(6, cols);
    kernel.set_arg(7, scale_size);
    kernel.set_arg(8, norm.eps);
    kernel.launch2d(kLocal, static_cast<std::size_t>(rows), kLocal, 1);
    autograd::record_op("rmsnorm_residual_add_scale_rowwise_wg_f32",
                        {residual.id(), x.id(), norm.weight.data.id(), scale.id()},
                        {out.id()});
    return out;
}

Tensor ModernSelfAttention::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0], true);
}

Tensor ModernSelfAttention::forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal) {
    MCL_CHECK(x.ndim() == 2 && x.shape()[1] == n_embd_, "ModernSelfAttention expects [B*T, n_embd]");
    QKV split = project_qkv(x);
    Tensor q;
    Tensor k;
    if (use_qk_norm_ &&
        can_use_fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                          head_dim_, batch_size, seq_len, use_rope_, rotary_dim_)) {
        auto fused = fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                               head_dim_, batch_size, 0, rope_theta_, rotary_dim_);
        q = std::move(fused.first);
        k = std::move(fused.second);
    } else {
        apply_qk_norm_if_enabled(*this, split.q, split.k);
        q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.q;
        k = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.k;
    }
    auto context = attention_window_ > 0
        ? motifcl::grouped_query_attention_windowed(q, k, split.v, n_head_, n_kv_head_, attention_window_,
                                                    causal, batch_size, seq_len, seq_len, 0)
        : motifcl::grouped_query_attention(q, k, split.v, n_head_, n_kv_head_, causal, batch_size, seq_len, seq_len, 0);
    auto out = o_proj_.forward(context);
    return dropout_p_ > 0.0f ? motifcl::dropout(out, dropout_p_, true) : out;
}

Tensor ModernSelfAttention::forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len, bool causal) {
    MCL_CHECK(x.ndim() == 2 && x.shape()[1] == n_embd_, "ModernSelfAttention expects [B*T, n_embd]");
    QKV split = project_qkv(x);
    Tensor q;
    Tensor k;
    if (use_qk_norm_ &&
        can_use_fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                          head_dim_, batch_size, seq_len, use_rope_, rotary_dim_)) {
        auto fused = fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                               head_dim_, batch_size, 0, rope_theta_, rotary_dim_);
        q = std::move(fused.first);
        k = std::move(fused.second);
    } else {
        apply_qk_norm_if_enabled(*this, split.q, split.k);
        q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.q;
        k = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.k;
    }
    auto context = motifcl::grouped_query_attention_masked(q, k, split.v, mask, n_head_, n_kv_head_,
                                                           causal, batch_size, seq_len, seq_len, 0, false);
    auto out = o_proj_.forward(context);
    return dropout_p_ > 0.0f ? motifcl::dropout(out, dropout_p_, true) : out;
}

Tensor ModernSelfAttention::forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(cache.k.valid() && cache.v.valid(), "ModernSelfAttention cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim, "ModernSelfAttention cache shape mismatch");
    MCL_CHECK(cache.length + seq_len <= cache.max_seq_len, "ModernSelfAttention KV cache capacity exceeded");
    QKV split = project_qkv(x);
    const int64_t offset = cache.length;
    Tensor q;
    if (!use_qk_norm_ &&
        can_use_fused_rope_cache_append_decode(split.q, split.k, split.v, cache.k, cache.v,
                                               n_head_, n_kv_head_, head_dim_,
                                               batch_size, seq_len, cache.max_seq_len, offset,
                                               use_rope_, rotary_dim_)) {
        q = fused_rope_cache_append_decode(split.q, split.k, split.v, cache.k, cache.v,
                                           n_head_, n_kv_head_, head_dim_,
                                           batch_size, cache.max_seq_len, offset,
                                           rope_theta_, rotary_dim_);
    } else if (use_qk_norm_ &&
        can_use_fused_qk_norm_rope_cache_append_decode(split.q, split.k, split.v, q_norm_, k_norm_,
                                                       cache.k, cache.v, n_head_, n_kv_head_, head_dim_,
                                                       batch_size, seq_len, cache.max_seq_len, offset,
                                                       use_rope_, rotary_dim_)) {
        q = fused_qk_norm_rope_cache_append_decode(split.q, split.k, split.v, q_norm_, k_norm_,
                                                   cache.k, cache.v, n_head_, n_kv_head_, head_dim_,
                                                   batch_size, cache.max_seq_len, offset,
                                                   rope_theta_, rotary_dim_);
    } else {
        Tensor k_new;
        if (use_qk_norm_ &&
            can_use_fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                              head_dim_, batch_size, seq_len, use_rope_, rotary_dim_)) {
            auto fused = fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                                   head_dim_, batch_size, offset, rope_theta_, rotary_dim_);
            q = std::move(fused.first);
            k_new = std::move(fused.second);
        } else {
            apply_qk_norm_if_enabled(*this, split.q, split.k);
            q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.q;
            k_new = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.k;
        }
        motifcl::kv_cache_append(k_new, split.v, cache.k, cache.v, batch_size, seq_len, cache.max_seq_len, offset);
    }
    cache.length += seq_len;
    const int64_t key_len = cache.length;
    auto context = attention_window_ > 0
        ? motifcl::grouped_query_attention_windowed(q, cache.k, cache.v, n_head_, n_kv_head_, attention_window_, true,
                                                    batch_size, seq_len, key_len, offset)
        : motifcl::grouped_query_attention(q, cache.k, cache.v, n_head_, n_kv_head_, true,
                                           batch_size, seq_len, key_len, offset);
    return o_proj_.forward(context);
}

Tensor ModernSelfAttention::forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(cache.k.valid() && cache.v.valid(), "ModernSelfAttention cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim, "ModernSelfAttention cache shape mismatch");
    MCL_CHECK(cache.length + seq_len <= cache.max_seq_len, "ModernSelfAttention KV cache capacity exceeded");
    QKV split = project_qkv(x);
    const int64_t offset = cache.length;
    Tensor q;
    Tensor k_new;
    if (use_qk_norm_ &&
        can_use_fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                          head_dim_, batch_size, seq_len, use_rope_, rotary_dim_)) {
        auto fused = fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                               head_dim_, batch_size, offset, rope_theta_, rotary_dim_);
        q = std::move(fused.first);
        k_new = std::move(fused.second);
    } else {
        apply_qk_norm_if_enabled(*this, split.q, split.k);
        q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.q;
        k_new = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.k;
    }
    motifcl::kv_cache_append(k_new, split.v, cache.k, cache.v, batch_size, seq_len, cache.max_seq_len, offset);
    cache.length += seq_len;
    auto context = motifcl::grouped_query_attention_masked(q, cache.k, cache.v, mask, n_head_, n_kv_head_, true,
                                                           batch_size, seq_len, cache.max_seq_len, offset, false);
    return o_proj_.forward(context);
}

Tensor ModernSelfAttention::forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                                                KVCache& cache, int64_t batch_size, int64_t seq_len,
                                                                int64_t cache_length_after, bool causal) {
    MCL_CHECK(cache.k.valid() && cache.v.valid(), "ModernSelfAttention cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim, "ModernSelfAttention cache shape mismatch");
    MCL_CHECK(cache_length_after >= cache.length && cache_length_after <= cache.max_seq_len, "ModernSelfAttention positioned KV cache length out of range");
    QKV split = project_qkv(x);
    apply_qk_norm_if_enabled(*this, split.q, split.k);
    auto q = use_rope_ ? motifcl::rope_positions(split.q, positions, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_) : split.q;
    auto k_new = use_rope_ ? motifcl::rope_positions(split.k, positions, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_) : split.k;
    motifcl::kv_cache_append_positions(k_new, split.v, positions, cache.k, cache.v, batch_size, seq_len, cache.max_seq_len);
    cache.length = cache_length_after;
    auto context = motifcl::grouped_query_attention_masked(q, cache.k, cache.v, mask, n_head_, n_kv_head_, causal,
                                                           batch_size, seq_len, cache.max_seq_len, 0, false);
    return o_proj_.forward(context);
}

Tensor ModernSelfAttention::forward_with_cache(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(cache.k_pages.valid() && cache.v_pages.valid() && cache.page_table.valid(),
              "ModernSelfAttention paged cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim,
              "ModernSelfAttention paged cache shape mismatch");
    MCL_CHECK(seq_len > 0 && seq_len <= cache.max_seq_len, "ModernSelfAttention invalid paged cache sequence length");
    QKV split = project_qkv(x);
    const int64_t query_abs_start = cache.tokens_seen;
    Tensor q;
    Tensor k_new;
    if (use_qk_norm_ &&
        can_use_fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                          head_dim_, batch_size, seq_len, use_rope_, rotary_dim_)) {
        auto fused = fused_qk_norm_rope_decode(split.q, split.k, q_norm_, k_norm_, n_head_, n_kv_head_,
                                               head_dim_, batch_size, query_abs_start, rope_theta_, rotary_dim_);
        q = std::move(fused.first);
        k_new = std::move(fused.second);
    } else {
        apply_qk_norm_if_enabled(*this, split.q, split.k);
        q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, query_abs_start) : split.q;
        k_new = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, query_abs_start) : split.k;
    }
    motifcl::paged_kv_cache_append(k_new, split.v, cache.page_table, cache.k_pages, cache.v_pages,
                                   batch_size, seq_len, cache.page_size, cache.page_count, query_abs_start);
    cache.tokens_seen += seq_len;
    cache.length = std::min<int64_t>(cache.max_seq_len, cache.tokens_seen);
    const int64_t key_len = cache.length;
    const int64_t key_abs_start = cache.tokens_seen - key_len;
    auto context = motifcl::paged_grouped_query_attention(q, cache.k_pages, cache.v_pages, cache.page_table,
                                                          n_head_, n_kv_head_, attention_window_, true,
                                                          batch_size, seq_len, key_len,
                                                          query_abs_start, key_abs_start,
                                                          cache.page_size, cache.page_count);
    return o_proj_.forward(context);
}

std::vector<Parameter*> ModernSelfAttention::parameters() {
    std::vector<Parameter*> result;
    if (use_split_projections_) {
        for (auto* layer : {&q_proj_, &k_proj_, &v_proj_}) {
            auto p = layer->parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
    } else {
        auto p = qkv_proj_.parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto p2 = o_proj_.parameters();
    result.insert(result.end(), p2.begin(), p2.end());
    if (use_qk_norm_) {
        auto qp = q_norm_.parameters();
        result.insert(result.end(), qp.begin(), qp.end());
        auto kp = k_norm_.parameters();
        result.insert(result.end(), kp.begin(), kp.end());
    }
    return result;
}

void ModernSelfAttention::enable_quantized_inference(DType qdtype) {
    if (use_split_projections_) {
        q_proj_.enable_quantized_inference(qdtype);
        k_proj_.enable_quantized_inference(qdtype);
        v_proj_.enable_quantized_inference(qdtype);
    } else {
        qkv_proj_.enable_quantized_inference(qdtype);
    }
    o_proj_.enable_quantized_inference(qdtype);
}

void ModernSelfAttention::disable_quantized_inference() {
    qkv_proj_.disable_quantized_inference();
    q_proj_.disable_quantized_inference();
    k_proj_.disable_quantized_inference();
    v_proj_.disable_quantized_inference();
    o_proj_.disable_quantized_inference();
}

bool ModernSelfAttention::quantized_inference_enabled() const {
    const bool qkv_quantized =
        use_split_projections_
            ? (q_proj_.quantized_inference_enabled() && k_proj_.quantized_inference_enabled() &&
               v_proj_.quantized_inference_enabled())
            : qkv_proj_.quantized_inference_enabled();
    return qkv_quantized && o_proj_.quantized_inference_enabled();
}

DType ModernSelfAttention::quantized_weight_dtype() const {
    if (!quantized_inference_enabled()) return DType::F32;
    return use_split_projections_ ? q_proj_.quantized_weight_dtype() : qkv_proj_.quantized_weight_dtype();
}

void ModernSelfAttention::set_attention_window(int window) {
    MCL_CHECK(window >= 0, "ModernSelfAttention attention window must be non-negative");
    attention_window_ = window;
}

GatedAttentionLayer::GatedAttentionLayer(Backend& backend, const TransformerConfig& raw_config)
    : attention(backend, raw_config),
      gate_proj(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, false) {}

Tensor GatedAttentionLayer::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0], true);
}

Tensor GatedAttentionLayer::forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal) {
    auto y = attention.forward(x, batch_size, seq_len, causal);
    return sigmoid_gate_mul(y, gate_proj.forward(x));
}

Tensor GatedAttentionLayer::forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    auto y = attention.forward_with_cache(x, cache, batch_size, seq_len);
    return sigmoid_gate_mul(y, gate_proj.forward(x));
}

Tensor GatedAttentionLayer::forward_with_cache(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len) {
    auto y = attention.forward_with_cache(x, cache, batch_size, seq_len);
    return sigmoid_gate_mul(y, gate_proj.forward(x));
}

std::vector<Parameter*> GatedAttentionLayer::parameters() {
    auto result = attention.parameters();
    auto gate_params = gate_proj.parameters();
    result.insert(result.end(), gate_params.begin(), gate_params.end());
    return result;
}

HybridTransformerBlock::HybridTransformerBlock(Backend& backend,
                                               const TransformerConfig& raw_config,
                                               const HybridLayerConfig& raw_layer_config)
    : layer_config(raw_layer_config),
      norm1_(backend, normalize_config(raw_config).n_embd),
      norm2_(backend, normalize_config(raw_config).n_embd) {
    auto cfg = normalize_config(raw_config);
    causal_ = cfg.causal;
    cfg.split_qkv_projections = true;
    cfg.split_mlp_projections = true;

    if (layer_config.attention == HybridAttentionKind::SlidingAttention) {
        cfg.sliding_window = layer_config.sliding_window > 0 ? layer_config.sliding_window : cfg.sliding_window;
        MCL_CHECK(cfg.sliding_window > 0, "Hybrid sliding-attention layer requires a positive sliding_window");
        attention_ = std::make_unique<ModernSelfAttention>(backend, cfg);
    } else if (layer_config.attention == HybridAttentionKind::GatedAttention) {
        cfg.sliding_window = layer_config.sliding_window > 0 ? layer_config.sliding_window : cfg.sliding_window;
        gated_attention_ = std::make_unique<GatedAttentionLayer>(backend, cfg);
        gated_attention_->attention.enable_split_projections(true);
    } else if (layer_config.attention == HybridAttentionKind::GatedDeltaNet) {
        cfg.sliding_window = 0;
        delta_ = std::make_unique<GatedDeltaNetLayer>(backend, cfg, layer_config.delta_state_dim);
    } else {
        cfg.sliding_window = 0;
        attention_ = std::make_unique<ModernSelfAttention>(backend, cfg);
    }
    if (attention_) attention_->enable_split_projections(true);

    if (layer_config.ffn == HybridFFNKind::MoEFFN) {
        const int experts = layer_config.num_experts;
        const int top_k = layer_config.experts_per_token;
        MCL_CHECK(experts > 0 && top_k > 0, "Hybrid MoE FFN requires num_experts and experts_per_token");
        moe_ = std::make_unique<MoEFFN>(backend, cfg.n_embd, cfg.mlp_hidden, experts, top_k);
    } else {
        mlp_ = std::make_unique<ModernMLP>(backend, cfg.n_embd, cfg.mlp_hidden,
                                           cfg.use_swiglu, cfg.use_qkv_bias, cfg.dropout);
        mlp_->enable_split_projections(true);
    }
}

Tensor HybridTransformerBlock::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0]);
}

Tensor HybridTransformerBlock::forward(const Tensor& x, int64_t batch_size, int64_t seq_len) {
    return forward_with_cache(x, nullptr, nullptr, nullptr, false, batch_size, seq_len);
}

Tensor HybridTransformerBlock::forward_with_cache(const Tensor& x,
                                                  KVCache* kv_cache,
                                                  PagedKVCache* paged_kv_cache,
                                                  DeltaStateCache* delta_state,
                                                  bool use_paged_kv,
                                                  int64_t batch_size,
                                                  int64_t seq_len) {
    MCL_CHECK(x.ndim() == 2, "HybridTransformerBlock expects flattened [tokens, channels] input");
    MCL_CHECK(batch_size > 0 && seq_len > 0 && batch_size * seq_len == x.shape()[0],
              "HybridTransformerBlock invalid batch/sequence dimensions");

    const auto normed = norm1_.forward(x);
    Tensor attn_out;
    if (delta_) {
        if (delta_state) {
            attn_out = delta_->forward_with_state(normed, *delta_state, batch_size, seq_len);
        } else {
            DeltaStateCache local_state(x.backend(), batch_size, delta_->n_head(), delta_->head_dim(), delta_->state_dim());
            attn_out = delta_->forward_with_state(normed, local_state, batch_size, seq_len);
        }
    } else if (gated_attention_) {
        if (use_paged_kv) {
            MCL_CHECK(paged_kv_cache != nullptr, "Hybrid gated-attention paged KV cache is missing");
            attn_out = gated_attention_->forward_with_cache(normed, *paged_kv_cache, batch_size, seq_len);
        } else if (kv_cache) {
            attn_out = gated_attention_->forward_with_cache(normed, *kv_cache, batch_size, seq_len);
        } else {
            attn_out = gated_attention_->forward(normed, batch_size, seq_len, causal_);
        }
    } else {
        MCL_CHECK(attention_ != nullptr, "Hybrid attention layer is not initialized");
        if (use_paged_kv) {
            MCL_CHECK(paged_kv_cache != nullptr, "Hybrid attention paged KV cache is missing");
            attn_out = attention_->forward_with_cache(normed, *paged_kv_cache, batch_size, seq_len);
        } else if (kv_cache) {
            attn_out = attention_->forward_with_cache(normed, *kv_cache, batch_size, seq_len);
        } else {
            attn_out = attention_->forward(normed, batch_size, seq_len, causal_);
        }
    }

    auto h = add(x, attn_out);
    auto ffn_in = norm2_.forward(h);
    Tensor ffn_out = moe_ ? moe_->forward(ffn_in) : mlp_->forward(ffn_in);
    return add(h, ffn_out);
}

std::vector<Parameter*> HybridTransformerBlock::parameters() {
    std::vector<Parameter*> result;
    auto add_params = [&](Module* module) {
        if (!module) return;
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    };
    add_params(&norm1_);
    add_params(attention_.get());
    add_params(gated_attention_.get());
    add_params(delta_.get());
    add_params(&norm2_);
    add_params(mlp_.get());
    add_params(moe_.get());
    return result;
}

bool HybridTransformerBlock::uses_kv_cache() const {
    return attention_ != nullptr || gated_attention_ != nullptr;
}

bool HybridTransformerBlock::uses_state_cache() const {
    return delta_ != nullptr;
}

ModernTransformerBlock::ModernTransformerBlock(Backend& backend, const TransformerConfig& raw_config)
    : norm1_(backend, normalize_config(raw_config).n_embd),
      attn_(backend, normalize_config(raw_config)),
      norm2_(backend, normalize_config(raw_config).n_embd),
      post_attention_norm_(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).rms_norm_eps),
      post_ffw_norm_(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).rms_norm_eps),
      layer_output_scale_(Tensor::ones(backend, {1})),
      mlp_(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).mlp_hidden,
           normalize_config(raw_config).use_swiglu, normalize_config(raw_config).use_qkv_bias,
           normalize_config(raw_config).dropout, normalize_config(raw_config).skip_weight_init) {
    const auto cfg = normalize_config(raw_config);
    dropout_p_ = cfg.dropout;
    causal_ = cfg.causal;
    use_post_attention_norm_ = cfg.use_post_attention_norm;
    use_post_ffw_norm_ = cfg.use_post_ffw_norm;
    use_layer_output_scale_ = cfg.use_layer_output_scale;
    use_per_layer_input_ = cfg.use_per_layer_inputs;
    if (use_per_layer_input_) {
        per_layer_input_gate_ = std::make_unique<Linear>(backend, cfg.n_embd, cfg.per_layer_input_dim,
                                                         false, cfg.skip_weight_init);
        per_layer_projection_ = std::make_unique<Linear>(backend, cfg.per_layer_input_dim, cfg.n_embd,
                                                         false, cfg.skip_weight_init);
        post_per_layer_norm_ = std::make_unique<RMSNorm>(backend, cfg.n_embd, cfg.rms_norm_eps);
    }
    if (cfg.split_qkv_projections) attn_.enable_split_projections(true);
    if (cfg.split_mlp_projections) mlp_.enable_split_projections(true);
}

Tensor ModernTransformerBlock::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0]);
}

Tensor ModernTransformerBlock::forward(const Tensor& x, int64_t batch_size, int64_t seq_len, const Tensor* per_layer_input) {
    auto attn_out = attn_.forward(norm1_.forward(x), batch_size, seq_len, causal_);
    auto h = apply_attention_residual(x, attn_out);
    h = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    h = apply_per_layer_input(h, per_layer_input);
    return apply_layer_output_scale(h);
}

Tensor ModernTransformerBlock::forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len,
                                              const Tensor* per_layer_input) {
    auto attn_out = attn_.forward_masked(norm1_.forward(x), mask, batch_size, seq_len, causal_);
    auto h = apply_attention_residual(x, attn_out);
    h = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    h = apply_per_layer_input(h, per_layer_input);
    return apply_layer_output_scale(h);
}

Tensor ModernTransformerBlock::forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len,
                                                  const Tensor* per_layer_input) {
    auto attn_out = attn_.forward_with_cache(norm1_.forward(x), cache, batch_size, seq_len);
    auto h = apply_attention_residual(x, attn_out);
    h = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    h = apply_per_layer_input(h, per_layer_input);
    return apply_layer_output_scale(h);
}

Tensor ModernTransformerBlock::forward_with_cache_packed_per_layer_input(const Tensor& x,
                                                                         KVCache& cache,
                                                                         int64_t batch_size,
                                                                         int64_t seq_len,
                                                                         const Tensor* packed_per_layer_input,
                                                                         int layer_index,
                                                                         int64_t token_count) {
    return forward_with_cache_packed_per_layer_input_replay(x, cache, batch_size, seq_len,
                                                            packed_per_layer_input, layer_index, token_count);
}

Tensor ModernTransformerBlock::forward_with_cache_packed_per_layer_input_eager(const Tensor& x,
                                                                               KVCache& cache,
                                                                               int64_t batch_size,
                                                                               int64_t seq_len,
                                                                               const Tensor* packed_per_layer_input,
                                                                               int layer_index,
                                                                               int64_t token_count) {
    auto attn_out = attn_.forward_with_cache(norm1_.forward(x), cache, batch_size, seq_len);
    auto h = apply_attention_residual(x, attn_out);
    return decode_tail_after_attention_packed_replay(h, packed_per_layer_input, layer_index, token_count);
}

Tensor ModernTransformerBlock::forward_with_cache_packed_per_layer_input_replay(const Tensor& x,
                                                                                KVCache& kv_cache,
                                                                                int64_t batch_size,
                                                                                int64_t seq_len,
                                                                                const Tensor* packed_per_layer_input,
                                                                                int layer_index,
                                                                                int64_t token_count) {
    const int64_t offset = kv_cache.length;
    const bool eligible =
        decode_full_block_graph_replay_enabled() &&
        !autograd::is_enabled() &&
        !autograd::is_graph_capturing() &&
        use_per_layer_input_ &&
        x.valid() &&
        x.dtype() == DType::F32 &&
        x.ndim() == 2 &&
        x.shape()[0] == 1 &&
        batch_size == 1 &&
        seq_len == 1 &&
        token_count == 1 &&
        kv_cache.k.valid() &&
        kv_cache.v.valid() &&
        offset >= 0 &&
        offset + seq_len <= kv_cache.max_seq_len &&
        packed_per_layer_input != nullptr &&
        packed_per_layer_input->valid() &&
        packed_per_layer_input->dtype() == DType::F32 &&
        packed_per_layer_input->backend_ptr() == x.backend_ptr() &&
        kv_cache.k.backend_ptr() == x.backend_ptr() &&
        kv_cache.v.backend_ptr() == x.backend_ptr();
    if (!eligible) {
        return forward_with_cache_packed_per_layer_input_eager(x, kv_cache, batch_size, seq_len,
                                                               packed_per_layer_input, layer_index, token_count);
    }

    const std::string key = std::to_string(layer_index) + ":" + std::to_string(offset) + ":" +
                            std::to_string(batch_size) + ":" + std::to_string(seq_len);
    std::lock_guard<std::mutex> lock(g_decode_full_block_graph_mutex);
    auto& by_key = g_decode_full_block_graph_cache[this];
    auto& cache = by_key[key];
    const bool cache_matches =
        cache.ready &&
        cache.executor &&
        cache.layer_index == layer_index &&
        cache.offset == offset &&
        cache.batch_size == batch_size &&
        cache.seq_len == seq_len &&
        cache.token_count == token_count &&
        cache.input_shape == x.shape() &&
        cache.packed_shape == packed_per_layer_input->shape() &&
        cache.cache_k_shape == kv_cache.k.shape() &&
        cache.cache_v_shape == kv_cache.v.shape();
    if (cache_matches) {
        Tensor out = Tensor::empty(x.backend(), cache.output_shape, cache.output_dtype);
        try {
            cache.executor->clear_bindings();
            cache.executor->bind_tensor(cache.input_id, x);
            cache.executor->bind_tensor(cache.packed_id, *packed_per_layer_input);
            cache.executor->bind_tensor(cache.cache_k_id, kv_cache.k);
            cache.executor->bind_tensor(cache.cache_v_id, kv_cache.v);
            cache.executor->bind_tensor(cache.output_id, out);
            cache.executor->execute();
            kv_cache.length = offset + seq_len;
            return out;
        } catch (const std::exception& e) {
            kv_cache.length = offset;
            cache.capture_failed = true;
            cache.ready = false;
            cache.executor.reset();
            if (decode_block_graph_replay_log_enabled()) {
                std::fprintf(stderr,
                             "[motifcl] full decode block replay disabled for layer=%d offset=%lld after failure: %s\n",
                             layer_index,
                             static_cast<long long>(offset),
                             e.what());
            }
            return forward_with_cache_packed_per_layer_input_eager(x, kv_cache, batch_size, seq_len,
                                                                   packed_per_layer_input, layer_index, token_count);
        }
    }
    if (cache.capture_failed) {
        return forward_with_cache_packed_per_layer_input_eager(x, kv_cache, batch_size, seq_len,
                                                               packed_per_layer_input, layer_index, token_count);
    }

    autograd::GraphCaptureGuard guard;
    autograd::register_tensor(x.id(), x.shape(), x.dtype(), x.nbytes(), x.buffer().raw());
    autograd::register_tensor(packed_per_layer_input->id(),
                              packed_per_layer_input->shape(),
                              packed_per_layer_input->dtype(),
                              packed_per_layer_input->nbytes(),
                              packed_per_layer_input->buffer().raw());
    autograd::register_tensor(kv_cache.k.id(), kv_cache.k.shape(), kv_cache.k.dtype(), kv_cache.k.nbytes(),
                              kv_cache.k.buffer().raw());
    autograd::register_tensor(kv_cache.v.id(), kv_cache.v.shape(), kv_cache.v.dtype(), kv_cache.v.nbytes(),
                              kv_cache.v.buffer().raw());
    Tensor out = forward_with_cache_packed_per_layer_input_eager(x, kv_cache, batch_size, seq_len,
                                                                 packed_per_layer_input, layer_index, token_count);
    auto graph = guard.finish();

    const bool usable_graph =
        !graph.empty() &&
        graph.replayable() &&
        graph.rebindable() &&
        graph.tensor_specs().find(x.id()) != graph.tensor_specs().end() &&
        graph.tensor_specs().find(packed_per_layer_input->id()) != graph.tensor_specs().end() &&
        graph.tensor_specs().find(kv_cache.k.id()) != graph.tensor_specs().end() &&
        graph.tensor_specs().find(kv_cache.v.id()) != graph.tensor_specs().end() &&
        graph.tensor_specs().find(out.id()) != graph.tensor_specs().end();
    if (!usable_graph) {
        cache.capture_failed = true;
        if (decode_block_graph_replay_log_enabled()) {
            std::fprintf(stderr,
                         "[motifcl] full decode block graph unavailable layer=%d offset=%lld nodes=%zu replayable=%d rebindable=%d\n",
                         layer_index,
                         static_cast<long long>(offset),
                         graph.size(),
                         graph.replayable() ? 1 : 0,
                         graph.rebindable() ? 1 : 0);
        }
        return out;
    }

    std::size_t kernel_count = 0;
    for (const auto& node : graph.nodes()) kernel_count += node.kernel_launches().size();

    try {
        autograd::GraphOptimizeOptions options;
        options.enable_buffer_reuse = true;
        auto executor = std::make_unique<autograd::GraphExecutor>(std::move(graph), options);
        cache.input_id = x.id();
        cache.packed_id = packed_per_layer_input->id();
        cache.cache_k_id = kv_cache.k.id();
        cache.cache_v_id = kv_cache.v.id();
        cache.output_id = out.id();
        cache.input_shape = x.shape();
        cache.packed_shape = packed_per_layer_input->shape();
        cache.cache_k_shape = kv_cache.k.shape();
        cache.cache_v_shape = kv_cache.v.shape();
        cache.output_shape = out.shape();
        cache.output_dtype = out.dtype();
        cache.layer_index = layer_index;
        cache.offset = offset;
        cache.batch_size = batch_size;
        cache.seq_len = seq_len;
        cache.token_count = token_count;
        cache.node_count = executor->graph().size();
        cache.kernel_count = kernel_count;
        cache.executor = std::move(executor);
        cache.ready = true;
        cache.capture_failed = false;
        if (decode_block_graph_replay_log_enabled()) {
            std::fprintf(stderr,
                         "[motifcl] captured full decode block graph layer=%d offset=%lld nodes=%zu kernels=%zu mode=%s arena=%zu\n",
                         layer_index,
                         static_cast<long long>(offset),
                         cache.node_count,
                         cache.kernel_count,
                         cache.executor->execution_mode().c_str(),
                         cache.executor->arena_bytes());
        }
    } catch (const std::exception& e) {
        cache.capture_failed = true;
        cache.ready = false;
        cache.executor.reset();
        if (decode_block_graph_replay_log_enabled()) {
            std::fprintf(stderr,
                         "[motifcl] full decode block graph capture failed layer=%d offset=%lld: %s\n",
                         layer_index,
                         static_cast<long long>(offset),
                         e.what());
        }
    }
    return out;
}

Tensor ModernTransformerBlock::forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache,
                                                         int64_t batch_size, int64_t seq_len,
                                                         const Tensor* per_layer_input) {
    auto attn_out = attn_.forward_with_cache_masked(norm1_.forward(x), mask, cache, batch_size, seq_len);
    auto h = apply_attention_residual(x, attn_out);
    h = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    h = apply_per_layer_input(h, per_layer_input);
    return apply_layer_output_scale(h);
}

Tensor ModernTransformerBlock::forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                                                   KVCache& cache, int64_t batch_size, int64_t seq_len,
                                                                   int64_t cache_length_after, bool causal,
                                                                   const Tensor* per_layer_input) {
    auto attn_out = attn_.forward_with_cache_positions_masked(norm1_.forward(x), positions, mask, cache,
                                                              batch_size, seq_len, cache_length_after, causal);
    auto h = apply_attention_residual(x, attn_out);
    h = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    h = apply_per_layer_input(h, per_layer_input);
    return apply_layer_output_scale(h);
}

Tensor ModernTransformerBlock::forward_with_cache(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len,
                                                  const Tensor* per_layer_input) {
    auto attn_out = attn_.forward_with_cache(norm1_.forward(x), cache, batch_size, seq_len);
    auto h = apply_attention_residual(x, attn_out);
    h = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    h = apply_per_layer_input(h, per_layer_input);
    return apply_layer_output_scale(h);
}

Tensor ModernTransformerBlock::forward_with_cache_packed_per_layer_input(const Tensor& x,
                                                                         PagedKVCache& cache,
                                                                         int64_t batch_size,
                                                                         int64_t seq_len,
                                                                         const Tensor* packed_per_layer_input,
                                                                         int layer_index,
                                                                         int64_t token_count) {
    auto attn_out = attn_.forward_with_cache(norm1_.forward(x), cache, batch_size, seq_len);
    auto h = apply_attention_residual(x, attn_out);
    return decode_tail_after_attention_packed_replay(h, packed_per_layer_input, layer_index, token_count);
}

Tensor ModernTransformerBlock::decode_tail_after_attention_packed(const Tensor& h,
                                                                  const Tensor* packed_per_layer_input,
                                                                  int layer_index,
                                                                  int64_t token_count) {
    Tensor out = (!use_post_ffw_norm_ && !use_per_layer_input_ && !use_layer_output_scale_)
        ? fused_or_eager_mlp_residual(h, norm2_, mlp_)
        : apply_ffn_residual(h, mlp_forward_rmsnorm_decode(h, norm2_, mlp_));
    return apply_packed_per_layer_input_and_scale(out, packed_per_layer_input, layer_index, token_count);
}

Tensor ModernTransformerBlock::decode_tail_after_attention_packed_replay(const Tensor& h,
                                                                         const Tensor* packed_per_layer_input,
                                                                         int layer_index,
                                                                         int64_t token_count) {
    const bool eligible =
        decode_block_graph_replay_enabled() &&
        !autograd::is_enabled() &&
        !autograd::is_graph_capturing() &&
        use_per_layer_input_ &&
        h.valid() &&
        h.dtype() == DType::F32 &&
        h.ndim() == 2 &&
        h.shape()[0] == 1 &&
        token_count == 1 &&
        packed_per_layer_input != nullptr &&
        packed_per_layer_input->valid() &&
        packed_per_layer_input->dtype() == DType::F32 &&
        packed_per_layer_input->backend_ptr() == h.backend_ptr();
    if (!eligible) {
        return decode_tail_after_attention_packed(h, packed_per_layer_input, layer_index, token_count);
    }

    std::lock_guard<std::mutex> lock(g_decode_block_tail_graph_mutex);
    auto& cache = g_decode_block_tail_graph_cache[this];
    const bool cache_matches =
        cache.ready &&
        cache.executor &&
        cache.layer_index == layer_index &&
        cache.input_shape == h.shape() &&
        cache.packed_shape == packed_per_layer_input->shape();
    if (cache_matches) {
        Tensor out = Tensor::empty(h.backend(), cache.output_shape, cache.output_dtype);
        try {
            cache.executor->clear_bindings();
            cache.executor->bind_tensor(cache.input_id, h);
            cache.executor->bind_tensor(cache.packed_id, *packed_per_layer_input);
            cache.executor->bind_tensor(cache.output_id, out);
            cache.executor->execute();
            return out;
        } catch (const std::exception& e) {
            cache.capture_failed = true;
            cache.ready = false;
            cache.executor.reset();
            if (decode_block_graph_replay_log_enabled()) {
                std::fprintf(stderr,
                             "[motifcl] decode block graph replay disabled for layer %d after replay failure: %s\n",
                             layer_index,
                             e.what());
            }
            return decode_tail_after_attention_packed(h, packed_per_layer_input, layer_index, token_count);
        }
    }
    if (cache.capture_failed) {
        return decode_tail_after_attention_packed(h, packed_per_layer_input, layer_index, token_count);
    }

    autograd::GraphCaptureGuard guard;
    autograd::register_tensor(h.id(), h.shape(), h.dtype(), h.nbytes(), h.buffer().raw());
    autograd::register_tensor(packed_per_layer_input->id(),
                              packed_per_layer_input->shape(),
                              packed_per_layer_input->dtype(),
                              packed_per_layer_input->nbytes(),
                              packed_per_layer_input->buffer().raw());
    Tensor out = decode_tail_after_attention_packed(h, packed_per_layer_input, layer_index, token_count);
    auto graph = guard.finish();

    const bool usable_graph =
        !graph.empty() &&
        graph.replayable() &&
        graph.rebindable() &&
        graph.tensor_specs().find(h.id()) != graph.tensor_specs().end() &&
        graph.tensor_specs().find(packed_per_layer_input->id()) != graph.tensor_specs().end() &&
        graph.tensor_specs().find(out.id()) != graph.tensor_specs().end();
    if (!usable_graph) {
        cache.capture_failed = true;
        if (decode_block_graph_replay_log_enabled()) {
            std::fprintf(stderr,
                         "[motifcl] decode block graph replay unavailable for layer %d: nodes=%zu replayable=%d rebindable=%d\n",
                         layer_index,
                         graph.size(),
                         graph.replayable() ? 1 : 0,
                         graph.rebindable() ? 1 : 0);
        }
        return out;
    }

    std::size_t kernel_count = 0;
    for (const auto& node : graph.nodes()) kernel_count += node.kernel_launches().size();

    try {
        autograd::GraphOptimizeOptions options;
        options.enable_buffer_reuse = true;
        auto executor = std::make_unique<autograd::GraphExecutor>(std::move(graph), options);
        cache.input_id = h.id();
        cache.packed_id = packed_per_layer_input->id();
        cache.output_id = out.id();
        cache.input_shape = h.shape();
        cache.packed_shape = packed_per_layer_input->shape();
        cache.output_shape = out.shape();
        cache.output_dtype = out.dtype();
        cache.layer_index = layer_index;
        cache.node_count = executor->graph().size();
        cache.kernel_count = kernel_count;
        cache.executor = std::move(executor);
        cache.ready = true;
        cache.capture_failed = false;
        if (decode_block_graph_replay_log_enabled()) {
            std::fprintf(stderr,
                         "[motifcl] captured decode block tail graph layer=%d nodes=%zu kernels=%zu mode=%s arena=%zu\n",
                         layer_index,
                         cache.node_count,
                         cache.kernel_count,
                         cache.executor->execution_mode().c_str(),
                         cache.executor->arena_bytes());
        }
    } catch (const std::exception& e) {
        cache.capture_failed = true;
        cache.ready = false;
        cache.executor.reset();
        if (decode_block_graph_replay_log_enabled()) {
            std::fprintf(stderr,
                         "[motifcl] decode block graph capture failed for layer %d: %s\n",
                         layer_index,
                         e.what());
        }
    }
    return out;
}

Tensor ModernTransformerBlock::apply_attention_residual(const Tensor& x, const Tensor& attn_out) {
    if (use_post_attention_norm_ && can_use_fused_rmsnorm_residual_add(x, attn_out, post_attention_norm_)) {
        return fused_rmsnorm_residual_add(x, attn_out, post_attention_norm_);
    }
    const auto y = use_post_attention_norm_ ? post_attention_norm_.forward(attn_out) : attn_out;
    return add(x, y);
}

Tensor ModernTransformerBlock::apply_ffn_residual(const Tensor& h, const Tensor& mlp_out) {
    if (use_post_ffw_norm_ && can_use_fused_rmsnorm_residual_add(h, mlp_out, post_ffw_norm_)) {
        return fused_rmsnorm_residual_add(h, mlp_out, post_ffw_norm_);
    }
    const auto y = use_post_ffw_norm_ ? post_ffw_norm_.forward(mlp_out) : mlp_out;
    return add(h, y);
}

Tensor ModernTransformerBlock::apply_per_layer_input(const Tensor& h, const Tensor* per_layer_input) {
    if (!use_per_layer_input_) return h;
    MCL_CHECK(per_layer_input != nullptr && per_layer_input->valid(),
              "ModernTransformerBlock expected a valid per-layer input tensor");
    MCL_CHECK(per_layer_input_gate_ && per_layer_projection_ && post_per_layer_norm_,
              "ModernTransformerBlock per-layer input modules are not initialized");
    MCL_CHECK(per_layer_input->shape().ndim() == 2 &&
                  per_layer_input->shape()[0] == h.shape()[0] &&
                  per_layer_input->shape()[1] == per_layer_input_gate_->out_features(),
              "ModernTransformerBlock per-layer input shape mismatch");
    auto mixed = gelu_mul_fast(per_layer_input_gate_->forward(h), *per_layer_input);
    auto projected = per_layer_projection_->forward(mixed);
    if (can_use_fused_rmsnorm_residual_add(h, projected, *post_per_layer_norm_)) {
        return fused_rmsnorm_residual_add(h, projected, *post_per_layer_norm_);
    }
    return add(h, post_per_layer_norm_->forward(projected));
}

Tensor ModernTransformerBlock::apply_packed_per_layer_input(const Tensor& h,
                                                            const Tensor* packed_per_layer_input,
                                                            int layer_index,
                                                            int64_t token_count) {
    if (!use_per_layer_input_) return h;
    MCL_CHECK(packed_per_layer_input != nullptr && packed_per_layer_input->valid(),
              "ModernTransformerBlock expected a valid packed per-layer input tensor");
    MCL_CHECK(per_layer_input_gate_ && per_layer_projection_ && post_per_layer_norm_,
              "ModernTransformerBlock per-layer input modules are not initialized");
    MCL_CHECK(token_count == h.shape()[0],
              "ModernTransformerBlock packed per-layer token count mismatch");
    MCL_CHECK(packed_per_layer_input->shape().ndim() == 2 &&
                  packed_per_layer_input->shape()[0] % token_count == 0 &&
                  packed_per_layer_input->shape()[1] == per_layer_input_gate_->out_features(),
              "ModernTransformerBlock packed per-layer input shape mismatch");
    if (can_use_fused_packed_ple_m1(h,
                                    *packed_per_layer_input,
                                    *per_layer_input_gate_,
                                    *per_layer_projection_,
                                    *post_per_layer_norm_,
                                    layer_index,
                                    token_count)) {
        return fused_packed_ple_m1(h,
                                   *packed_per_layer_input,
                                   *per_layer_input_gate_,
                                   *per_layer_projection_,
                                   *post_per_layer_norm_,
                                   layer_index);
    }
    auto mixed = can_use_fused_packed_ple_gate_m1(h,
                                                 *packed_per_layer_input,
                                                 *per_layer_input_gate_,
                                                 layer_index,
                                                 token_count)
        ? fused_packed_ple_gate_m1(h, *packed_per_layer_input, *per_layer_input_gate_, layer_index)
        : gelu_mul_packed_per_layer_fast(per_layer_input_gate_->forward(h),
                                         *packed_per_layer_input,
                                         layer_index,
                                         token_count);
    auto projected = per_layer_projection_->forward(mixed);
    if (can_use_fused_rmsnorm_residual_add(h, projected, *post_per_layer_norm_)) {
        return fused_rmsnorm_residual_add(h, projected, *post_per_layer_norm_);
    }
    return add(h, post_per_layer_norm_->forward(projected));
}

Tensor ModernTransformerBlock::apply_packed_per_layer_input_and_scale(const Tensor& h,
                                                                      const Tensor* packed_per_layer_input,
                                                                      int layer_index,
                                                                      int64_t token_count) {
    if (!use_per_layer_input_) return apply_layer_output_scale(h);
    MCL_CHECK(packed_per_layer_input != nullptr && packed_per_layer_input->valid(),
              "ModernTransformerBlock expected a valid packed per-layer input tensor");
    MCL_CHECK(per_layer_input_gate_ && per_layer_projection_ && post_per_layer_norm_,
              "ModernTransformerBlock per-layer input modules are not initialized");
    MCL_CHECK(token_count == h.shape()[0],
              "ModernTransformerBlock packed per-layer token count mismatch");
    MCL_CHECK(packed_per_layer_input->shape().ndim() == 2 &&
                  packed_per_layer_input->shape()[0] % token_count == 0 &&
                  packed_per_layer_input->shape()[1] == per_layer_input_gate_->out_features(),
              "ModernTransformerBlock packed per-layer input shape mismatch");
    auto mixed = can_use_fused_packed_ple_gate_m1(h,
                                                 *packed_per_layer_input,
                                                 *per_layer_input_gate_,
                                                 layer_index,
                                                 token_count)
        ? fused_packed_ple_gate_m1(h, *packed_per_layer_input, *per_layer_input_gate_, layer_index)
        : gelu_mul_packed_per_layer_fast(per_layer_input_gate_->forward(h),
                                         *packed_per_layer_input,
                                         layer_index,
                                         token_count);
    auto projected = per_layer_projection_->forward(mixed);
    if (use_layer_output_scale_ &&
        can_use_fused_rmsnorm_residual_add_scale(h, projected, *post_per_layer_norm_, layer_output_scale_.data)) {
        return fused_rmsnorm_residual_add_scale(h, projected, *post_per_layer_norm_, layer_output_scale_.data);
    }
    Tensor out = can_use_fused_rmsnorm_residual_add(h, projected, *post_per_layer_norm_)
        ? fused_rmsnorm_residual_add(h, projected, *post_per_layer_norm_)
        : add(h, post_per_layer_norm_->forward(projected));
    return apply_layer_output_scale(out);
}

Tensor ModernTransformerBlock::apply_layer_output_scale(const Tensor& h) {
    return use_layer_output_scale_ ? motifcl::mul(h, layer_output_scale_.data) : h;
}

std::vector<Parameter*> ModernTransformerBlock::parameters() {
    std::vector<Parameter*> result;
    for (auto* module : {static_cast<Module*>(&norm1_), static_cast<Module*>(&attn_), static_cast<Module*>(&norm2_), static_cast<Module*>(&mlp_)}) {
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    if (use_post_attention_norm_) {
        auto p = post_attention_norm_.parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    if (use_post_ffw_norm_) {
        auto p = post_ffw_norm_.parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    if (use_layer_output_scale_) result.push_back(&layer_output_scale_);
    if (use_per_layer_input_) {
        for (auto* module : {static_cast<Module*>(per_layer_input_gate_.get()),
                             static_cast<Module*>(per_layer_projection_.get()),
                             static_cast<Module*>(post_per_layer_norm_.get())}) {
            if (!module) continue;
            auto p = module->parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
    }
    return result;
}

void ModernTransformerBlock::enable_quantized_inference(DType qdtype) {
    attn_.enable_quantized_inference(qdtype);
    mlp_.enable_quantized_inference(qdtype);
}

void ModernTransformerBlock::disable_quantized_inference() {
    attn_.disable_quantized_inference();
    mlp_.disable_quantized_inference();
}

bool ModernTransformerBlock::quantized_inference_enabled() const {
    return attn_.quantized_inference_enabled() && mlp_.quantized_inference_enabled();
}

DType ModernTransformerBlock::quantized_weight_dtype() const {
    return quantized_inference_enabled() ? attn_.quantized_weight_dtype() : DType::F32;
}

ModernGPTModel::ModernGPTModel(Backend& backend, const TransformerConfig& raw_config)
    : config(normalize_config(raw_config)),
      token_embedding(backend, normalize_config(raw_config).vocab_size, normalize_config(raw_config).n_embd,
                      normalize_config(raw_config).skip_weight_init),
      position_embedding(Tensor::randn(backend,
                                       {normalize_config(raw_config).learned_position_embeddings
                                            ? normalize_config(raw_config).block_size
                                            : 1,
                                        normalize_config(raw_config).n_embd},
                                       0.02f),
                         normalize_config(raw_config).learned_position_embeddings),
      final_norm(backend, normalize_config(raw_config).n_embd),
      lm_head(random_parameter_or_empty(backend,
                                        {normalize_config(raw_config).n_embd,
                                         normalize_config(raw_config).vocab_size},
                                        0.02f,
                                        normalize_config(raw_config).skip_weight_init)) {
    use_positions_ = config.learned_position_embeddings && !config.use_rope;
    if (config.use_per_layer_inputs) {
        per_layer_token_embedding = std::make_unique<Embedding>(
            backend,
            config.per_layer_input_vocab_size > 0 ? config.per_layer_input_vocab_size : config.vocab_size,
            config.n_layer * config.per_layer_input_dim,
            config.skip_weight_init);
        per_layer_model_projection = std::make_unique<Linear>(
            backend, config.n_embd, config.n_layer * config.per_layer_input_dim, false, config.skip_weight_init);
        per_layer_projection_norm = std::make_unique<RMSNorm>(
            backend, config.per_layer_input_dim, config.rms_norm_eps);
    }
    blocks.reserve(static_cast<std::size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i) {
        auto layer_config = config;
        if (!config.layer_head_dims.empty()) {
            layer_config.head_dim = config.layer_head_dims[static_cast<std::size_t>(i)];
            if (layer_config.rotary_dim <= 0 || layer_config.rotary_dim > layer_config.head_dim) {
                layer_config.rotary_dim = layer_config.head_dim;
            }
        }
        if (!config.layer_mlp_hiddens.empty()) {
            layer_config.mlp_hidden = config.layer_mlp_hiddens[static_cast<std::size_t>(i)];
        }
        if (!config.layer_rope_thetas.empty()) {
            layer_config.rope_theta = config.layer_rope_thetas[static_cast<std::size_t>(i)];
        }
        blocks.push_back(std::make_shared<ModernTransformerBlock>(backend, layer_config));
    }
}

Tensor ModernGPTModel::input_embeddings(const Tensor& token_ids, int64_t batch_size, int64_t seq_len) {
    Tensor h = use_positions_
        ? token_position_embedding(token_ids, token_embedding.weight.data, position_embedding.data)
        : token_embedding.forward(token_ids).view({batch_size * seq_len, config.n_embd});
    if (config.embedding_scale != 1.0f) h = motifcl::scale(h, config.embedding_scale);
    return h;
}

Tensor ModernGPTModel::compute_per_layer_inputs(const Tensor& token_ids,
                                                const Tensor& inputs_embeds,
                                                int64_t batch_size,
                                                int64_t seq_len) {
    MCL_CHECK(config.use_per_layer_inputs, "ModernGPTModel PLE is not enabled");
    MCL_CHECK(per_layer_token_embedding && per_layer_model_projection && per_layer_projection_norm,
              "ModernGPTModel PLE modules are not initialized");
    const int64_t token_count = batch_size * seq_len;
    const int64_t total_dim = static_cast<int64_t>(config.n_layer) * config.per_layer_input_dim;
    auto token_identity = per_layer_token_embedding->forward(token_ids).view({token_count, total_dim});
    token_identity = motifcl::scale(token_identity, std::sqrt(static_cast<float>(config.per_layer_input_dim)));

    auto projected = per_layer_model_projection->forward(inputs_embeds);
    projected = motifcl::scale(projected, 1.0f / std::sqrt(static_cast<float>(config.n_embd)));
    projected = per_layer_projection_norm->forward(projected.view({token_count * config.n_layer,
                                                                   config.per_layer_input_dim}))
                    .view({token_count, total_dim});
    return motifcl::scale(add(projected, token_identity), 0.7071067811865476f)
        .view({token_count * config.n_layer, config.per_layer_input_dim});
}

Tensor ModernGPTModel::per_layer_input_slice(const Tensor& packed, int layer, int64_t token_count) const {
    MCL_CHECK(layer >= 0 && layer < config.n_layer, "ModernGPTModel PLE layer index out of range");
    MCL_CHECK(packed.dtype() == DType::F32 && packed.ndim() == 2,
              "ModernGPTModel PLE packed input must be f32 rank-2");
    MCL_CHECK(packed.shape()[0] == token_count * config.n_layer &&
                  packed.shape()[1] == config.per_layer_input_dim,
              "ModernGPTModel PLE packed input shape mismatch");
    auto out = Tensor::empty(packed.backend(), {token_count, config.per_layer_input_dim}, DType::F32);
    auto k = packed.backend().kernels.get("embedding_per_layer_slice_f32");
    const int n = static_cast<int>(token_count * config.per_layer_input_dim);
    k.set_arg(0, packed.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, static_cast<int>(token_count));
    k.set_arg(3, config.n_layer);
    k.set_arg(4, config.per_layer_input_dim);
    k.set_arg(5, layer);
    k.set_arg(6, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("embedding_per_layer_slice_f32", {packed.id()}, {out.id()});
    return out;
}

Tensor ModernGPTModel::forward(const Tensor& token_ids) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T <= config.block_size, "ModernGPTModel sequence length exceeds block_size");
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        Tensor layer_input = config.use_per_layer_inputs
            ? per_layer_input_slice(per_layer_inputs, static_cast<int>(i), B * T)
            : Tensor{};
        h = blocks[i]->forward(h, B, T, config.use_per_layer_inputs ? &layer_input : nullptr);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_masked(const Tensor& token_ids, const Tensor& mask) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T <= config.block_size, "ModernGPTModel sequence length exceeds block_size");
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        Tensor layer_input = config.use_per_layer_inputs
            ? per_layer_input_slice(per_layer_inputs, static_cast<int>(i), B * T)
            : Tensor{};
        h = blocks[i]->forward_masked(h, mask, B, T, config.use_per_layer_inputs ? &layer_input : nullptr);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache(const Tensor& token_ids, std::vector<KVCache>& caches) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache(token_ids, ptrs);
}

Tensor ModernGPTModel::forward_with_cache(const Tensor& token_ids, std::vector<KVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        const int64_t start = caches[0]->length;
        MCL_CHECK(start + T <= config.block_size, "ModernGPTModel KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == start, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = config.use_per_layer_inputs
            ? blocks[i]->forward_with_cache_packed_per_layer_input(h, *caches[i], B, T,
                                                                   &per_layer_inputs,
                                                                   static_cast<int>(i),
                                                                   B * T)
            : blocks[i]->forward_with_cache(h, *caches[i], B, T, nullptr);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache_last_logits(const Tensor& token_ids, std::vector<KVCache>& caches) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache_last_logits(token_ids, ptrs);
}

Tensor ModernGPTModel::forward_with_cache_last_logits(const Tensor& token_ids, std::vector<KVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        const int64_t start = caches[0]->length;
        MCL_CHECK(start + T <= config.block_size, "ModernGPTModel KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == start, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = config.use_per_layer_inputs
            ? blocks[i]->forward_with_cache_packed_per_layer_input(h, *caches[i], B, T,
                                                                   &per_layer_inputs,
                                                                   static_cast<int>(i),
                                                                   B * T)
            : blocks[i]->forward_with_cache(h, *caches[i], B, T, nullptr);
    }
    h = final_norm.forward(h);
    return project_logits(last_sequence_hidden_rows(h, B, T, config.n_embd));
}

Tensor ModernGPTModel::forward_with_cache(const Tensor& token_ids, std::vector<PagedKVCache>& caches) {
    std::vector<PagedKVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache(token_ids, ptrs);
}

Tensor ModernGPTModel::forward_with_cache(const Tensor& token_ids, std::vector<PagedKVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel paged KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel paged cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid paged cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null paged KV cache");
        const int64_t start = caches[0]->tokens_seen;
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null paged KV cache");
            MCL_CHECK(cache->tokens_seen == start, "ModernGPTModel paged KV caches must have equal current length");
        }
    }
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = config.use_per_layer_inputs
            ? blocks[i]->forward_with_cache_packed_per_layer_input(h, *caches[i], B, T,
                                                                   &per_layer_inputs,
                                                                   static_cast<int>(i),
                                                                   B * T)
            : blocks[i]->forward_with_cache(h, *caches[i], B, T, nullptr);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache_last_logits(const Tensor& token_ids, std::vector<PagedKVCache>& caches) {
    std::vector<PagedKVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache_last_logits(token_ids, ptrs);
}

Tensor ModernGPTModel::forward_with_cache_last_logits(const Tensor& token_ids, std::vector<PagedKVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel paged KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel paged cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid paged cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null paged KV cache");
        const int64_t start = caches[0]->tokens_seen;
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null paged KV cache");
            MCL_CHECK(cache->tokens_seen == start, "ModernGPTModel paged KV caches must have equal current length");
        }
    }
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = config.use_per_layer_inputs
            ? blocks[i]->forward_with_cache_packed_per_layer_input(h, *caches[i], B, T,
                                                                   &per_layer_inputs,
                                                                   static_cast<int>(i),
                                                                   B * T)
            : blocks[i]->forward_with_cache(h, *caches[i], B, T, nullptr);
    }
    h = final_norm.forward(h);
    return project_logits(last_sequence_hidden_rows(h, B, T, config.n_embd));
}

Tensor ModernGPTModel::decode_step(const Tensor& token_ids, std::vector<KVCache>& caches) {
    check_decode_step_token_shape(token_ids, "ModernGPTModel");
    return forward_with_cache_last_logits(token_ids, caches);
}

Tensor ModernGPTModel::decode_step(const Tensor& token_ids, std::vector<KVCache*>& caches) {
    check_decode_step_token_shape(token_ids, "ModernGPTModel");
    return forward_with_cache_last_logits(token_ids, caches);
}

Tensor ModernGPTModel::decode_step(const Tensor& token_ids, std::vector<PagedKVCache>& caches) {
    check_decode_step_token_shape(token_ids, "ModernGPTModel");
    return forward_with_cache_last_logits(token_ids, caches);
}

Tensor ModernGPTModel::decode_step(const Tensor& token_ids, std::vector<PagedKVCache*>& caches) {
    check_decode_step_token_shape(token_ids, "ModernGPTModel");
    return forward_with_cache_last_logits(token_ids, caches);
}

Tensor ModernGPTModel::forward_with_cache_masked(const Tensor& token_ids, const Tensor& mask, std::vector<KVCache>& caches) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache_masked(token_ids, mask, ptrs);
}

Tensor ModernGPTModel::forward_with_cache_masked(const Tensor& token_ids, const Tensor& mask, std::vector<KVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        const int64_t start = caches[0]->length;
        MCL_CHECK(start + T <= config.block_size, "ModernGPTModel KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == start, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        Tensor layer_input = config.use_per_layer_inputs
            ? per_layer_input_slice(per_layer_inputs, static_cast<int>(i), B * T)
            : Tensor{};
        h = blocks[i]->forward_with_cache_masked(h, mask, *caches[i], B, T,
                                                 config.use_per_layer_inputs ? &layer_input : nullptr);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache_positions_masked(const Tensor& token_ids, const Tensor& positions, const Tensor& mask,
                                                              std::vector<KVCache>& caches, int64_t cache_length_after,
                                                              bool causal) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache_positions_masked(token_ids, positions, mask, ptrs, cache_length_after, causal);
}

Tensor ModernGPTModel::forward_with_cache_positions_masked(const Tensor& token_ids, const Tensor& positions, const Tensor& mask,
                                                              std::vector<KVCache*>& caches, int64_t cache_length_after,
                                                              bool causal) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(positions.dtype() == DType::I32 && positions.ndim() == 2, "ModernGPTModel positioned cache expects i32 [B,T] positions");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    MCL_CHECK(positions.shape()[0] == B && positions.shape()[1] == T, "ModernGPTModel positions shape mismatch");
    MCL_CHECK(positions.backend_ptr() == token_ids.backend_ptr(), "ModernGPTModel positions must share backend with token ids");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        MCL_CHECK(cache_length_after >= caches[0]->length && cache_length_after <= config.block_size, "ModernGPTModel positioned KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == caches[0]->length, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = input_embeddings(token_ids, B, T);
    Tensor per_layer_inputs = config.use_per_layer_inputs ? compute_per_layer_inputs(token_ids, h, B, T) : Tensor{};
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        Tensor layer_input = config.use_per_layer_inputs
            ? per_layer_input_slice(per_layer_inputs, static_cast<int>(i), B * T)
            : Tensor{};
        h = blocks[i]->forward_with_cache_positions_masked(h, positions, mask, *caches[i], B, T,
                                                           cache_length_after, causal,
                                                           config.use_per_layer_inputs ? &layer_input : nullptr);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

std::vector<Parameter*> ModernGPTModel::parameters() {
    auto result = token_embedding.parameters();
    if (use_positions_) result.push_back(&position_embedding);
    if (config.use_per_layer_inputs) {
        if (per_layer_token_embedding) {
            auto p = per_layer_token_embedding->parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
        if (per_layer_model_projection) {
            auto p = per_layer_model_projection->parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
        if (per_layer_projection_norm) {
            auto p = per_layer_projection_norm->parameters();
            result.insert(result.end(), p.begin(), p.end());
        }
    }
    for (auto& block : blocks) {
        auto p = block->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto nf = final_norm.parameters();
    result.insert(result.end(), nf.begin(), nf.end());
    result.push_back(&lm_head);
    return result;
}

std::vector<KVCache> ModernGPTModel::create_kv_cache(Backend& backend, int64_t batch_size) const {
    std::vector<KVCache> caches;
    caches.reserve(static_cast<std::size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i) {
        const auto& attn = blocks[static_cast<std::size_t>(i)]->attention();
        caches.emplace_back(backend, batch_size, config.block_size, attn.n_kv_head(), attn.head_dim());
    }
    return caches;
}

std::vector<PagedKVCache> ModernGPTModel::create_paged_kv_cache(Backend& backend,
                                                                int64_t batch_size,
                                                                int64_t page_size) const {
    std::vector<PagedKVCache> caches;
    caches.reserve(static_cast<std::size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i) {
        const auto& attn = blocks[static_cast<std::size_t>(i)]->attention();
        caches.emplace_back(backend, batch_size, config.block_size, page_size, attn.n_kv_head(), attn.head_dim());
    }
    return caches;
}

std::vector<DeltaStateCache> ModernGPTModel::create_delta_state_cache(Backend& backend,
                                                                      int64_t batch_size,
                                                                      int state_dim) const {
    std::vector<DeltaStateCache> caches;
    caches.reserve(static_cast<std::size_t>(config.n_layer));
    const int head_dim = config.head_dim;
    const int effective_state_dim = state_dim > 0 ? state_dim : head_dim;
    for (int i = 0; i < config.n_layer; ++i) {
        caches.emplace_back(backend, batch_size, config.n_head, head_dim, effective_state_dim);
    }
    return caches;
}

void ModernGPTModel::set_layer_attention_window(int layer, int window) {
    MCL_CHECK(layer >= 0 && layer < static_cast<int>(blocks.size()), "ModernGPTModel layer index out of range");
    MCL_CHECK(window >= 0, "ModernGPTModel attention window must be non-negative");
    blocks[static_cast<std::size_t>(layer)]->set_attention_window(window);
}

void ModernGPTModel::enable_quantized_inference(DType qdtype) {
    MCL_CHECK(qdtype == DType::Q8_0 || qdtype == DType::Q4_0,
              "ModernGPTModel quantized inference expects qdtype Q8_0 or Q4_0");
    for (auto& block : blocks) block->enable_quantized_inference(qdtype);
    quantized_weight_dtype_ = qdtype;
    quantized_lm_head_ = qdtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(lm_head.data)
        : quantize_q8_symmetric_cols(lm_head.data);
    decode_quantized_lm_head_ = Tensor{};
    decode_quantized_lm_head_dtype_ = DType::F32;
}

void ModernGPTModel::enable_quantized_inference(const QuantizationPolicy& policy) {
    MCL_CHECK(policy.default_dtype == DType::Q8_0 || policy.default_dtype == DType::Q4_0,
              "ModernGPTModel quantization policy default dtype must be Q8_0 or Q4_0");
    MCL_CHECK(policy.lm_head_dtype == DType::Q8_0 || policy.lm_head_dtype == DType::Q4_0,
              "ModernGPTModel quantization policy lm_head dtype must be Q8_0 or Q4_0");
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const bool use_q8 = std::find(policy.q8_layers.begin(), policy.q8_layers.end(), static_cast<int>(i)) != policy.q8_layers.end();
        blocks[i]->enable_quantized_inference(use_q8 ? DType::Q8_0 : policy.default_dtype);
    }
    quantized_weight_dtype_ = policy.default_dtype;
    quantized_lm_head_ = policy.lm_head_dtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(lm_head.data)
        : quantize_q8_symmetric_cols(lm_head.data);
    decode_quantized_lm_head_ = Tensor{};
    decode_quantized_lm_head_dtype_ = DType::F32;
}

void ModernGPTModel::set_quantized_lm_head(const Tensor& weight) {
    MCL_CHECK(weight.valid() && (weight.dtype() == DType::Q8_0 || weight.dtype() == DType::Q4_0 ||
                                 weight.dtype() == DType::Q4_0_COL || weight.dtype() == DType::Q4_K ||
                                 weight.dtype() == DType::Q5_K || weight.dtype() == DType::Q6_K),
              "ModernGPTModel quantized lm_head must be Q8_0, Q4_0, Q4_0_COL, Q4_K, Q5_K, or Q6_K");
    MCL_CHECK(weight.ndim() == 2 && weight.shape()[0] == config.n_embd && weight.shape()[1] == config.vocab_size,
              "ModernGPTModel quantized lm_head shape mismatch");
    quantized_lm_head_ = weight;
    quantized_weight_dtype_ = weight.dtype();
    decode_quantized_lm_head_ = Tensor{};
    decode_quantized_lm_head_dtype_ = DType::F32;
}

void ModernGPTModel::set_decode_quantized_lm_head(const Tensor& weight) {
    MCL_CHECK(weight.valid() && (weight.dtype() == DType::Q8_0 || weight.dtype() == DType::Q4_0 ||
                                 weight.dtype() == DType::Q4_0_COL || weight.dtype() == DType::Q4_K ||
                                 weight.dtype() == DType::Q5_K || weight.dtype() == DType::Q6_K),
              "ModernGPTModel decode quantized lm_head must be Q8_0, Q4_0, Q4_0_COL, Q4_K, Q5_K, or Q6_K");
    MCL_CHECK(weight.ndim() == 2 && weight.shape()[0] == config.n_embd && weight.shape()[1] == config.vocab_size,
              "ModernGPTModel decode quantized lm_head shape mismatch");
    decode_quantized_lm_head_ = weight;
    decode_quantized_lm_head_dtype_ = weight.dtype();
}

void ModernGPTModel::disable_quantized_inference() {
    for (auto& block : blocks) block->disable_quantized_inference();
    quantized_lm_head_ = Tensor{};
    decode_quantized_lm_head_ = Tensor{};
    quantized_weight_dtype_ = DType::F32;
    decode_quantized_lm_head_dtype_ = DType::F32;
}

Tensor ModernGPTModel::project_logits(const Tensor& h) {
    if (quantized_lm_head_.valid() && (!autograd::is_enabled() || !lm_head.data.valid())) {
        const Tensor& decode_head = decode_quantized_lm_head_.valid() ? decode_quantized_lm_head_ : quantized_lm_head_;
        const DType decode_dtype = decode_quantized_lm_head_.valid() ? decode_quantized_lm_head_dtype_ : quantized_weight_dtype_;
        if ((decode_dtype == DType::Q4_0 || decode_dtype == DType::Q4_0_COL ||
             is_k_quant_dtype(decode_dtype)) &&
            h.ndim() == 2 && h.shape()[0] == 1) {
            return matmul(h, decode_head);
        }
        auto hq = quantize_q8_symmetric_rows(h);
        return matmul(hq, quantized_lm_head_);
    }
    MCL_CHECK(lm_head.data.valid(), "ModernGPTModel dense lm_head is not initialized");
    return matmul(h, lm_head.data);
}

void HybridRuntimeCache::reset() {
    for (auto& cache : kv_caches) cache.reset();
    for (auto& cache : paged_kv_caches) cache.reset();
    for (auto& state : delta_states) state.zero();
}

HybridGPTModel::HybridGPTModel(Backend& backend,
                               const TransformerConfig& raw_config,
                               const std::vector<HybridLayerConfig>& raw_layer_configs)
    : config(normalize_config(raw_config)),
      token_embedding(backend, normalize_config(raw_config).vocab_size, normalize_config(raw_config).n_embd,
                      normalize_config(raw_config).skip_weight_init),
      position_embedding(Tensor::randn(backend,
                                       {normalize_config(raw_config).learned_position_embeddings
                                            ? normalize_config(raw_config).block_size
                                            : 1,
                                        normalize_config(raw_config).n_embd},
                                       0.02f),
                         normalize_config(raw_config).learned_position_embeddings),
      final_norm(backend, normalize_config(raw_config).n_embd),
      lm_head(random_parameter_or_empty(backend,
                                        {normalize_config(raw_config).n_embd,
                                         normalize_config(raw_config).vocab_size},
                                        0.02f,
                                        normalize_config(raw_config).skip_weight_init)) {
    use_positions_ = config.learned_position_embeddings && !config.use_rope;
    layer_configs.reserve(static_cast<std::size_t>(config.n_layer));
    blocks.reserve(static_cast<std::size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i) {
        HybridLayerConfig layer_cfg;
        if (i < static_cast<int>(raw_layer_configs.size())) layer_cfg = raw_layer_configs[static_cast<std::size_t>(i)];
        layer_configs.push_back(layer_cfg);
        blocks.push_back(std::make_shared<HybridTransformerBlock>(backend, config, layer_cfg));
    }
}

Tensor HybridGPTModel::forward(const Tensor& token_ids) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "HybridGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "HybridGPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "HybridGPTModel sequence length exceeds block_size");
    Tensor h = use_positions_
        ? token_position_embedding(token_ids, token_embedding.weight.data, position_embedding.data)
        : token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (auto& block : blocks) h = block->forward(h, B, T);
    h = final_norm.forward(h);
    auto logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor HybridGPTModel::forward_with_cache(const Tensor& token_ids, HybridRuntimeCache& cache) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "HybridGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "HybridGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(!use_positions_, "HybridGPTModel cached inference currently expects RoPE/token-only positions");
    MCL_CHECK(cache.kv_caches.size() == blocks.size() &&
                  cache.paged_kv_caches.size() == blocks.size() &&
                  cache.delta_states.size() == blocks.size(),
              "HybridGPTModel runtime cache count must match layer count");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "HybridGPTModel invalid cached sequence length");
    MCL_CHECK(cache.batch_size == B, "HybridGPTModel runtime cache batch size mismatch");

    int64_t first_kv_length = -1;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if (!blocks[i]->uses_kv_cache()) continue;
        const auto length = cache.use_paged_kv
            ? cache.paged_kv_caches[i].tokens_seen
            : cache.kv_caches[i].length;
        if (first_kv_length < 0) first_kv_length = length;
        MCL_CHECK(length == first_kv_length, "HybridGPTModel KV caches must have equal current length");
    }
    if (first_kv_length >= 0) {
        MCL_CHECK(first_kv_length + T <= config.block_size,
                  "HybridGPTModel KV cache exceeds block_size");
    }

    Tensor h = token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = blocks[i]->forward_with_cache(h,
                                          blocks[i]->uses_kv_cache() ? &cache.kv_caches[i] : nullptr,
                                          blocks[i]->uses_kv_cache() ? &cache.paged_kv_caches[i] : nullptr,
                                          blocks[i]->uses_state_cache() ? &cache.delta_states[i] : nullptr,
                                          cache.use_paged_kv,
                                          B,
                                          T);
    }
    h = final_norm.forward(h);
    auto logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor HybridGPTModel::forward_with_cache_last_logits(const Tensor& token_ids, HybridRuntimeCache& cache) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "HybridGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "HybridGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(!use_positions_, "HybridGPTModel cached inference currently expects RoPE/token-only positions");
    MCL_CHECK(cache.kv_caches.size() == blocks.size() &&
                  cache.paged_kv_caches.size() == blocks.size() &&
                  cache.delta_states.size() == blocks.size(),
              "HybridGPTModel runtime cache count must match layer count");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "HybridGPTModel invalid cached sequence length");
    MCL_CHECK(cache.batch_size == B, "HybridGPTModel runtime cache batch size mismatch");

    int64_t first_kv_length = -1;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if (!blocks[i]->uses_kv_cache()) continue;
        const auto length = cache.use_paged_kv
            ? cache.paged_kv_caches[i].tokens_seen
            : cache.kv_caches[i].length;
        if (first_kv_length < 0) first_kv_length = length;
        MCL_CHECK(length == first_kv_length, "HybridGPTModel KV caches must have equal current length");
    }
    if (first_kv_length >= 0) {
        MCL_CHECK(first_kv_length + T <= config.block_size,
                  "HybridGPTModel KV cache exceeds block_size");
    }

    Tensor h = token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = blocks[i]->forward_with_cache(h,
                                          blocks[i]->uses_kv_cache() ? &cache.kv_caches[i] : nullptr,
                                          blocks[i]->uses_kv_cache() ? &cache.paged_kv_caches[i] : nullptr,
                                          blocks[i]->uses_state_cache() ? &cache.delta_states[i] : nullptr,
                                          cache.use_paged_kv,
                                          B,
                                          T);
    }
    h = final_norm.forward(h);
    return project_logits(last_sequence_hidden_rows(h, B, T, config.n_embd));
}

Tensor HybridGPTModel::decode_step(const Tensor& token_ids, HybridRuntimeCache& cache) {
    check_decode_step_token_shape(token_ids, "HybridGPTModel");
    return forward_with_cache_last_logits(token_ids, cache);
}

std::vector<Parameter*> HybridGPTModel::parameters() {
    auto result = token_embedding.parameters();
    if (use_positions_) result.push_back(&position_embedding);
    for (auto& block : blocks) {
        auto p = block->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto nf = final_norm.parameters();
    result.insert(result.end(), nf.begin(), nf.end());
    result.push_back(&lm_head);
    return result;
}

HybridRuntimeCache HybridGPTModel::create_runtime_cache(Backend& backend,
                                                        int64_t batch_size,
                                                        bool use_paged_kv,
                                                        int64_t page_size) const {
    MCL_CHECK(batch_size > 0, "HybridGPTModel runtime cache batch size must be positive");
    MCL_CHECK(page_size > 0, "HybridGPTModel paged KV page_size must be positive");
    HybridRuntimeCache cache;
    cache.use_paged_kv = use_paged_kv;
    cache.batch_size = batch_size;
    cache.page_size = use_paged_kv ? page_size : 0;
    cache.kv_caches.reserve(blocks.size());
    cache.paged_kv_caches.reserve(blocks.size());
    cache.delta_states.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]->uses_kv_cache()) {
            const auto* attn = blocks[i]->attention();
            const auto* gated = blocks[i]->gated_attention();
            const auto& layer_attn = attn ? *attn : gated->attention;
            cache.kv_caches.emplace_back(backend, batch_size, config.block_size,
                                         layer_attn.n_kv_head(), layer_attn.head_dim());
            cache.paged_kv_caches.emplace_back(backend, batch_size, config.block_size, page_size,
                                               layer_attn.n_kv_head(), layer_attn.head_dim());
        } else {
            cache.kv_caches.emplace_back();
            cache.paged_kv_caches.emplace_back();
        }
        if (blocks[i]->uses_state_cache()) {
            const auto* delta = blocks[i]->delta();
            cache.delta_states.emplace_back(backend, batch_size, delta->n_head(), delta->head_dim(), delta->state_dim());
        } else {
            cache.delta_states.emplace_back();
        }
    }
    return cache;
}

void HybridGPTModel::set_quantized_lm_head(const Tensor& weight) {
    MCL_CHECK(weight.valid() && (weight.dtype() == DType::Q8_0 || weight.dtype() == DType::Q4_0 ||
                                  weight.dtype() == DType::Q4_0_COL || weight.dtype() == DType::Q4_K ||
                                  weight.dtype() == DType::Q5_K || weight.dtype() == DType::Q6_K),
              "HybridGPTModel quantized lm_head must be Q8_0, Q4_0, Q4_0_COL, Q4_K, Q5_K, or Q6_K");
    MCL_CHECK(weight.ndim() == 2 && weight.shape()[0] == config.n_embd && weight.shape()[1] == config.vocab_size,
              "HybridGPTModel quantized lm_head shape mismatch");
    quantized_lm_head_ = weight;
    quantized_weight_dtype_ = weight.dtype();
    decode_quantized_lm_head_ = Tensor{};
    decode_quantized_lm_head_dtype_ = DType::F32;
}

void HybridGPTModel::set_decode_quantized_lm_head(const Tensor& weight) {
    MCL_CHECK(weight.valid() && (weight.dtype() == DType::Q8_0 || weight.dtype() == DType::Q4_0 ||
                                  weight.dtype() == DType::Q4_0_COL || weight.dtype() == DType::Q4_K ||
                                  weight.dtype() == DType::Q5_K || weight.dtype() == DType::Q6_K),
              "HybridGPTModel decode quantized lm_head must be Q8_0, Q4_0, Q4_0_COL, Q4_K, Q5_K, or Q6_K");
    MCL_CHECK(weight.ndim() == 2 && weight.shape()[0] == config.n_embd && weight.shape()[1] == config.vocab_size,
              "HybridGPTModel decode quantized lm_head shape mismatch");
    decode_quantized_lm_head_ = weight;
    decode_quantized_lm_head_dtype_ = weight.dtype();
}

void HybridGPTModel::disable_quantized_inference() {
    quantized_lm_head_ = Tensor{};
    decode_quantized_lm_head_ = Tensor{};
    quantized_weight_dtype_ = DType::F32;
    decode_quantized_lm_head_dtype_ = DType::F32;
}

Tensor HybridGPTModel::project_logits(const Tensor& h) {
    if (quantized_lm_head_.valid() && (!autograd::is_enabled() || !lm_head.data.valid())) {
        const Tensor& decode_head = decode_quantized_lm_head_.valid() ? decode_quantized_lm_head_ : quantized_lm_head_;
        const DType decode_dtype = decode_quantized_lm_head_.valid() ? decode_quantized_lm_head_dtype_ : quantized_weight_dtype_;
        if ((decode_dtype == DType::Q4_0 || decode_dtype == DType::Q4_0_COL ||
             is_k_quant_dtype(decode_dtype)) &&
            h.ndim() == 2 && h.shape()[0] == 1) {
            return matmul(h, decode_head);
        }
        auto hq = quantize_q8_symmetric_rows(h);
        return matmul(hq, quantized_lm_head_);
    }
    MCL_CHECK(lm_head.data.valid(), "HybridGPTModel dense lm_head is not initialized");
    return matmul(h, lm_head.data);
}

} // namespace motifcl::nn
