// MN-MoE15M — sparsely-routed counter-synapse MoE pretrain runner on Vulkan (RX 580 class).
//
// Backbone: stories15M-shaped (dim 288, 6 layers, MHA-6, RoPE, RMSNorm, vocab
// 32k, seq 256), tied or untied LM head. The dense SwiGLU FFN is replaced by a
// seven-expert routed MoE (nn::MoE7FFN): every expert is a pair of
// CounterStateLinear synapses, so experts carry no FP32 master weights and no
// optimizer moments; a hard top-1 router (fog_hard_route7) selects one expert
// per token with a straight-through softmax gradient, plus one always-on shared
// counter expert. Load balancing is aux-loss-free: a host-side per-expert bias.
//
// Data: packed uint16 records of length seq+1 (BOS + sliding window), the same
// train.bin / val.bin the dense ternary15M runner uses.
//
// env: DATA_DIR, OUT_DIR (required); STEPS, BATCH, SEQ, LR, LR_FP, WARMUP_FRAC,
// MIN_LR_FRAC, EVAL_EVERY, EVAL_RECORDS, CKPT_EVERY, LOG_EVERY, SEED, RESUME,
// TIE_HEAD, HIDDEN_EXP (expert width), HIDDEN_SHARED, BALANCE_GAMMA, SAVE_CKPT.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/moe7.hpp>
#include <motifcl/serialization.hpp>

namespace {

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}
float env_float(const char* name, float fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::strtof(v, nullptr) : fallback;
}
std::string env_str(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string(fallback);
}

std::vector<std::uint16_t> load_records(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<std::uint16_t> data(static_cast<std::size_t>(bytes) / 2);
    f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(data.size() * 2));
    if (!f) throw std::runtime_error("short read " + path);
    return data;
}

struct FpParam {
    motifcl::Tensor* t;
    motifcl::Tensor m, v;
    float weight_decay;
};

} // namespace

int main() {
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "no Vulkan device: " << probe.error << "\n";
        return 77;
    }
    try {
        Backend backend = Backend::create_vulkan();
        auto& runtime = backend.vulkan_runtime();
        // Opt-in per-submission GPU timing (VK timestamp queries): separates GPU
        // execution time from CPU recording/dispatch overhead in the step log.
        const bool gpu_timing = env_int("GPU_TIMING", 0) != 0;
        runtime.set_gpu_timing_enabled(gpu_timing);

        const std::string data_dir = env_str("DATA_DIR", "");
        const std::string out_dir = env_str("OUT_DIR", "");
        if (data_dir.empty() || out_dir.empty()) {
            std::cerr << "DATA_DIR and OUT_DIR are required\n";
            return 1;
        }
        std::filesystem::create_directories(out_dir);

        const int steps_total = env_int("STEPS", 20000);
        const std::int64_t B = env_int("BATCH", 8);
        const std::int64_t T = env_int("SEQ", 256);
        const float lr_counter_peak = env_float("LR", 4e-3f);
        const float lr_fp_peak = env_float("LR_FP", 6e-3f);
        const float warmup_frac = env_float("WARMUP_FRAC", 0.02f);
        const float min_lr_frac = env_float("MIN_LR_FRAC", 0.1f);
        const int eval_every = env_int("EVAL_EVERY", 500);
        const int eval_records = env_int("EVAL_RECORDS", 96);
        const std::int64_t eval_batch =
            (std::int64_t)env_int("EVAL_BATCH", (int)std::min<std::int64_t>(32, B));
        const int ckpt_every = env_int("CKPT_EVERY", 1000);
        const int log_every = env_int("LOG_EVERY", 20);
        const unsigned base_seed = (unsigned)env_int("SEED", 1337);
        const bool resume = env_int("RESUME", 1) != 0;
        const bool tie_head = env_int("TIE_HEAD", 0) != 0;
        const bool save_ckpt_enabled = env_int("SAVE_CKPT", 1) != 0;
        const int hidden_exp = env_int("HIDDEN_EXP", 192);
        const int hidden_shared = env_int("HIDDEN_SHARED", 192);
        const float balance_gamma = env_float("BALANCE_GAMMA", 1e-2f);
        const float aux_weight = env_float("AUX_WEIGHT", 0.01f);
        const float balance_clamp = env_float("BALANCE_CLAMP", 2.0f);

        // Model shape is env-driven so the same trainer can serve the 15M study and a
        // ~1B-class memory-native MoE. Constraints checked below: dim % heads == 0,
        // head_dim even (RoPE), dim and all counter widths divisible by 4 (6-bit packing).
        const std::int64_t dim = env_int("DIM", 288);
        const std::int64_t vocab = 32000;
        const int n_head = env_int("HEADS", 6);
        const int n_kv_head = n_head;
        const int n_layers = env_int("LAYERS", 6);
        const int C = 11;
        MCL_CHECK(dim % n_head == 0, "DIM must be divisible by HEADS");
        MCL_CHECK(((dim / n_head) % 2) == 0, "head_dim must be even for RoPE");
        MCL_CHECK(dim % 4 == 0, "DIM must be divisible by 4 for 6-bit counter packing");
        MCL_CHECK(hidden_exp % 4 == 0 && hidden_shared % 4 == 0,
                  "expert widths must be divisible by 4");
        const std::int64_t N = B * T;
        const float eps = 1e-5f;
        // The packed corpus stores fixed 257-token records. REC_LEN is the record stride in
        // the file; SEQ (T) may be shorter than REC_LEN-1 to cut the activation footprint of a
        // large model without repacking the data.
        const std::int64_t rec_len = env_int("REC_LEN", T + 1);
        MCL_CHECK(rec_len >= T + 1, "REC_LEN must be at least SEQ+1");

        int start_step = 0;
        const std::string meta_path = out_dir + "/meta.txt";
        if (resume && std::filesystem::exists(meta_path)) {
            std::ifstream mf(meta_path);
            mf >> start_step;
            std::cout << "resuming from step " << start_step << "\n";
        }

        auto train = load_records(data_dir + "/train.bin");
        auto val = load_records(std::filesystem::exists(data_dir + "/val.bin")
                            ? data_dir + "/val.bin" : data_dir + "/validation.bin");
        const std::int64_t n_train = (std::int64_t)(train.size() / (std::size_t)rec_len);
        const std::int64_t n_val = (std::int64_t)(val.size() / (std::size_t)rec_len);
        if (n_train == 0 || n_val == 0) throw std::runtime_error("empty dataset");
        std::cout << "train records=" << n_train << " val records=" << n_val
                  << " tokens/step=" << N << "\n";

        // ---- model ----
        std::vector<float> emb_host((std::size_t)(vocab * dim));
        {
            std::mt19937 rng(base_seed);
            std::normal_distribution<float> gauss(0.0f, 0.02f);
            for (auto& x : emb_host) x = gauss(rng);
        }
        auto emb = Tensor::from_cpu(backend, {vocab, dim}, DType::F32, emb_host.data());
        emb.set_requires_grad(true);
        Tensor head;
        if (!tie_head) {
            std::vector<float> head_host((std::size_t)(dim * vocab));
            std::mt19937 rng(base_seed + 7u);
            std::normal_distribution<float> gauss(0.0f, 0.02f);
            for (auto& x : head_host) x = gauss(rng);
            head = Tensor::from_cpu(backend, {dim, vocab}, DType::F32, head_host.data());
            head.set_requires_grad(true);
        }
        const std::int64_t max_flat = std::max(N, (std::int64_t)32 * T);
        std::vector<float> pos_zero((std::size_t)(max_flat * dim), 0.0f);
        auto pos_w = Tensor::from_cpu(backend, {max_flat, dim}, DType::F32, pos_zero.data());

        struct Block {
            std::unique_ptr<nn::CounterStateLinear> wq, wk, wv, wo;
            Tensor norm1, norm2;
            std::unique_ptr<nn::MoE7FFN> moe;
        };
        std::vector<float> ones((std::size_t)dim, 1.0f);
        std::vector<Block> blocks;
        for (int l = 0; l < n_layers; ++l) {
            const unsigned s0 = 1000u * (unsigned)(l + 1) + (unsigned)start_step;
            Block blk{
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, lr_counter_peak, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 1),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, lr_counter_peak, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 2),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, lr_counter_peak, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 3),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, lr_counter_peak, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 4),
                Tensor::from_cpu(backend, {dim}, DType::F32, ones.data()),
                Tensor::from_cpu(backend, {dim}, DType::F32, ones.data()),
                nullptr,
            };
            blk.norm1.set_requires_grad(true);
            blk.norm2.set_requires_grad(true);
            nn::MoE7Config moe_cfg;
            moe_cfg.dim = (int)dim;
            moe_cfg.hidden_expert = hidden_exp;
            moe_cfg.hidden_shared = hidden_shared;
            moe_cfg.C = C;
            moe_cfg.counter_lr = lr_counter_peak;
            moe_cfg.balance_clamp = balance_clamp;
            moe_cfg.seed = s0 + 500000u;   // disjoint from the attention counters above
            blk.moe = std::make_unique<nn::MoE7FFN>(backend, moe_cfg);
            blocks.push_back(std::move(blk));
        }
        auto norm_f = Tensor::from_cpu(backend, {dim}, DType::F32, ones.data());
        norm_f.set_requires_grad(true);

        // ---- fp optimizer state (Adam) ----
        std::vector<FpParam> fp;
        auto add_fp = [&](Tensor* t, float wd) {
            FpParam p{t, Tensor::empty(backend, t->shape(), DType::F32),
                      Tensor::empty(backend, t->shape(), DType::F32), wd};
            std::vector<float> zeros((std::size_t)t->numel(), 0.0f);
            p.m = Tensor::from_cpu(backend, t->shape(), DType::F32, zeros.data());
            p.v = Tensor::from_cpu(backend, t->shape(), DType::F32, zeros.data());
            fp.push_back(std::move(p));
        };
        add_fp(&emb, 0.1f);
        if (!tie_head) add_fp(&head, 0.1f);
        for (auto& blk : blocks) {
            add_fp(&blk.norm1, 0.0f);
            add_fp(&blk.norm2, 0.0f);
            add_fp(&blk.moe->router_weight.data, 0.0f);
        }
        add_fp(&norm_f, 0.0f);

        // ---- checkpoint io ----
        auto ckpt_path = [&](const std::string& name) { return out_dir + "/" + name; };
        auto save_ckpt = [&](int step) {
            save_tensor(emb, ckpt_path("emb.mt"));
            if (!tie_head) save_tensor(head, ckpt_path("head.mt"));
            save_tensor(norm_f, ckpt_path("norm_f.mt"));
            for (int l = 0; l < n_layers; ++l) {
                auto& blk = blocks[(std::size_t)l];
                const std::string p = "l" + std::to_string(l) + "_";
                save_tensor(blk.norm1, ckpt_path(p + "n1.mt"));
                save_tensor(blk.norm2, ckpt_path(p + "n2.mt"));
                nn::CounterStateLinear* attn[4] = {blk.wq.get(), blk.wk.get(), blk.wv.get(), blk.wo.get()};
                const char* attn_nm[4] = {"wq", "wk", "wv", "wo"};
                for (int i = 0; i < 4; ++i) {
                    save_tensor(attn[i]->state, ckpt_path(p + attn_nm[i] + "_state.mt"));
                    save_tensor(attn[i]->scale, ckpt_path(p + attn_nm[i] + "_scale.mt"));
                    save_tensor(attn[i]->v, ckpt_path(p + attn_nm[i] + "_v.mt"));
                }
                save_tensor(blk.moe->router_weight.data, ckpt_path(p + "router.mt"));
                save_tensor(blk.moe->router_bias, ckpt_path(p + "router_bias.mt"));
                auto* counters = blk.moe->counter_layers().data();
                const int n_counters = (int)blk.moe->counter_layers().size();
                for (int i = 0; i < n_counters; ++i) {
                    nn::CounterStateLinear* layer = counters[i];
                    const std::string name = p + "moe" + std::to_string(i);
                    save_tensor(layer->state, ckpt_path(name + "_state.mt"));
                    save_tensor(layer->scale, ckpt_path(name + "_scale.mt"));
                    save_tensor(layer->v, ckpt_path(name + "_v.mt"));
                }
            }
            for (std::size_t i = 0; i < fp.size(); ++i) {
                save_tensor(fp[i].m, ckpt_path("adam_m_" + std::to_string(i) + ".mt"));
                save_tensor(fp[i].v, ckpt_path("adam_v_" + std::to_string(i) + ".mt"));
            }
            std::ofstream mf(meta_path);
            mf << step << "\n";
        };
        auto load_ckpt = [&]() {
            auto load_into = [&](Tensor& dst, const std::string& path) {
                auto t = load_tensor(backend, path);
                auto host = t.to_vector<float>();
                dst = Tensor::from_cpu(backend, t.shape(), DType::F32, host.data());
            };
            load_into(emb, ckpt_path("emb.mt"));
            emb.set_requires_grad(true);
            if (!tie_head) {
                load_into(head, ckpt_path("head.mt"));
                head.set_requires_grad(true);
            }
            load_into(norm_f, ckpt_path("norm_f.mt"));
            norm_f.set_requires_grad(true);
            for (int l = 0; l < n_layers; ++l) {
                auto& blk = blocks[(std::size_t)l];
                const std::string p = "l" + std::to_string(l) + "_";
                load_into(blk.norm1, ckpt_path(p + "n1.mt"));
                blk.norm1.set_requires_grad(true);
                load_into(blk.norm2, ckpt_path(p + "n2.mt"));
                blk.norm2.set_requires_grad(true);
                nn::CounterStateLinear* attn[4] = {blk.wq.get(), blk.wk.get(), blk.wv.get(), blk.wo.get()};
                const char* attn_nm[4] = {"wq", "wk", "wv", "wo"};
                for (int i = 0; i < 4; ++i) {
                    attn[i]->state = load_tensor(backend, ckpt_path(p + attn_nm[i] + "_state.mt"));
                    attn[i]->scale = load_tensor(backend, ckpt_path(p + attn_nm[i] + "_scale.mt"));
                    attn[i]->v = load_tensor(backend, ckpt_path(p + attn_nm[i] + "_v.mt"));
                }
                blk.moe->router_weight.data = load_tensor(backend, ckpt_path(p + "router.mt"));
                blk.moe->router_weight.data.set_requires_grad(true);
                blk.moe->router_bias = load_tensor(backend, ckpt_path(p + "router_bias.mt"));
                blk.moe->router_bias.set_requires_grad(false);
                auto counters = blk.moe->counter_layers();
                for (std::size_t i = 0; i < counters.size(); ++i) {
                    const std::string name = p + "moe" + std::to_string(i);
                    counters[i]->state = load_tensor(backend, ckpt_path(name + "_state.mt"));
                    counters[i]->scale = load_tensor(backend, ckpt_path(name + "_scale.mt"));
                    counters[i]->v = load_tensor(backend, ckpt_path(name + "_v.mt"));
                }
            }
            for (std::size_t i = 0; i < fp.size(); ++i) {
                fp[i].m = load_tensor(backend, ckpt_path("adam_m_" + std::to_string(i) + ".mt"));
                fp[i].v = load_tensor(backend, ckpt_path("adam_v_" + std::to_string(i) + ".mt"));
            }
            // Rebind fp param pointers: load_into replaced several tensors.
            std::size_t k = 0;
            fp[k++].t = &emb;
            if (!tie_head) fp[k++].t = &head;
            for (auto& blk : blocks) {
                fp[k++].t = &blk.norm1;
                fp[k++].t = &blk.norm2;
                fp[k++].t = &blk.moe->router_weight.data;
            }
            fp[k].t = &norm_f;
        };
        if (start_step > 0) load_ckpt();

        auto forward_logits = [&](const Tensor& ids, std::int64_t bs) {
            auto x = nn::token_position_embedding(ids, emb, pos_w);
            Tensor cur = x;
            for (auto& blk : blocks) {
                auto a = rmsnorm(cur, blk.norm1, eps);
                auto q = rope(blk.wq->forward(a), n_head, bs, T);
                auto k = rope(blk.wk->forward(a), n_kv_head, bs, T);
                auto v = blk.wv->forward(a);
                auto attn = grouped_query_attention(q, k, v, n_head, n_kv_head, true,
                                                    (int)bs, T, T, 0, 0.0f);
                auto o = blk.wo->forward(attn);
                auto h = add(cur, o);
                auto b2 = rmsnorm(h, blk.norm2, eps);
                auto m = blk.moe->forward(b2);
                cur = add(h, m);
            }
            auto xf = rmsnorm(cur, norm_f, eps);
            return tie_head ? matmul_transpose_b(xf, emb) : matmul(xf, head);
        };

        auto forward_loss = [&](const Tensor& ids, const Tensor& targets, std::int64_t bs) {
            return softmax_cross_entropy(forward_logits(ids, bs), targets);
        };

        auto upload_batch = [&](const std::vector<std::uint16_t>& pool, std::int64_t n_rec,
                                std::int64_t bs, unsigned seed, Tensor& ids, Tensor& tg,
                                bool sequential_from0) {
            std::vector<std::int32_t> ih((std::size_t)(bs * T)), th((std::size_t)(bs * T));
            std::mt19937 rng(seed);
            std::uniform_int_distribution<std::int64_t> pick(0, n_rec - 1);
            for (std::int64_t b = 0; b < bs; ++b) {
                const std::int64_t r = sequential_from0 ? b : pick(rng);
                const std::uint16_t* rec = pool.data() + (std::size_t)(r * rec_len);
                for (std::int64_t t = 0; t < T; ++t) {
                    ih[(std::size_t)(b * T + t)] = (std::int32_t)rec[t];
                    th[(std::size_t)(b * T + t)] = (std::int32_t)rec[t + 1];
                }
            }
            ids = Tensor::from_cpu(backend, {bs * T}, DType::I32, ih.data());
            tg = Tensor::from_cpu(backend, {bs * T}, DType::I32, th.data());
        };

        auto set_counter_lr = [&](float lr) {
            for (auto& blk : blocks) {
                for (auto* l : {blk.wq.get(), blk.wk.get(), blk.wv.get(), blk.wo.get()}) l->set_lr(lr);
                blk.moe->set_lr(lr);
            }
        };
        auto set_counter_training = [&](bool t) {
            for (auto& blk : blocks) {
                for (auto* l : {blk.wq.get(), blk.wk.get(), blk.wv.get(), blk.wo.get()}) l->set_training(t);
                blk.moe->set_training(t);
            }
        };

        auto eval_val = [&]() {
            set_counter_training(false);
            // Eval must not be heavier than training: a 32-record eval batch allocates
            // 4x the vocabulary-logit buffers of a B8 training step and pushed the
            // device high-water mark from 3.4 GB to 6.8 GB on the 8 GB card.
            const std::int64_t eb = std::max<std::int64_t>(1, eval_batch);
            const int chunks = std::max(1, eval_records / (int)eb);
            double acc = 0.0;
            for (int c = 0; c < chunks; ++c) {
                std::vector<std::uint16_t> window(
                    val.begin() + (std::ptrdiff_t)((std::int64_t)c * eb * rec_len),
                    val.begin() + (std::ptrdiff_t)(((std::int64_t)c + 1) * eb * rec_len));
                Tensor ids, tg;
                upload_batch(window, eb, eb, 0, ids, tg, true);
                auto loss = forward_loss(ids, tg, eb);
                acc += (double)loss.item();
            }
            set_counter_training(true);
            return (float)(acc / chunks);
        };

        const int warmup_steps = std::max(1, (int)(steps_total * warmup_frac));
        auto lr_at = [&](int step, float peak) {
            if (step < warmup_steps) return peak * (float)(step + 1) / (float)warmup_steps;
            const float t = (float)(step - warmup_steps) / (float)std::max(1, steps_total - warmup_steps);
            const float cosv = 0.5f * (1.0f + std::cos(3.14159265358979f * t));
            const float lo = peak * min_lr_frac;
            return lo + (peak - lo) * cosv;
        };

        // Evaluation-only mode: load the checkpoint named by meta.txt, then report
        // loss/PPL plus next-token top-1/top-5 accuracy over the same fixed validation
        // records the training loop uses. No training, no state updates.
        if (env_int("EVAL_ONLY", 0) != 0) {
            if (start_step == 0) {
                std::cerr << "EVAL_ONLY needs an existing checkpoint (meta.txt in OUT_DIR)\n";
                return 1;
            }
            set_counter_training(false);
            const std::int64_t eb = std::max<std::int64_t>(1, eval_batch);
            const int chunks = std::max(1, eval_records / (int)eb);
            const std::int64_t rows_per_chunk = eb * T;
            const bool route_hist = env_int("ROUTE_HIST", 0) != 0;
            constexpr int kExp = 7;
            static_assert(kExp == nn::MoE7FFN::kExperts, "expert count drifted from the module");
            std::array<std::vector<std::int64_t>, 7> hist{};
            std::array<std::int64_t, 7> expert_tokens{};
            if (route_hist) {
                for (auto& h : hist) h.assign((std::size_t)vocab, 0);
            }
            double loss_sum = 0.0;
            std::int64_t top1 = 0, top5 = 0, total = 0;
            for (int c = 0; c < chunks; ++c) {
                std::vector<std::int32_t> ih((std::size_t)rows_per_chunk), th((std::size_t)rows_per_chunk);
                const std::int64_t rec_base = (std::int64_t)c * eb;
                for (std::int64_t b = 0; b < eb; ++b) {
                    const std::uint16_t* rec = val.data() + (std::size_t)((rec_base + b) * rec_len);
                    for (std::int64_t t = 0; t < T; ++t) {
                        ih[(std::size_t)(b * T + t)] = (std::int32_t)rec[t];
                        th[(std::size_t)(b * T + t)] = (std::int32_t)rec[t + 1];
                    }
                }
                auto ids = Tensor::from_cpu(backend, {rows_per_chunk}, DType::I32, ih.data());
                auto tg = Tensor::from_cpu(backend, {rows_per_chunk}, DType::I32, th.data());
                auto logits = forward_logits(ids, eb);
                auto loss = softmax_cross_entropy(logits, tg);
                loss_sum += (double)loss.item();
                const auto host_logits = logits.to_vector<float>();
                MCL_CHECK(host_logits.size() == (std::size_t)(rows_per_chunk * vocab),
                          "EVAL_ONLY logits size mismatch");
                for (std::int64_t r = 0; r < rows_per_chunk; ++r) {
                    const float* row = host_logits.data() + (std::size_t)(r * vocab);
                    const int target = (int)th[(std::size_t)r];
                    const float neg_inf = -std::numeric_limits<float>::infinity();
                    float best_v = neg_inf;
                    int best_i = -1;
                    float top_v[5] = {neg_inf, neg_inf, neg_inf, neg_inf, neg_inf};
                    int top_i[5] = {-1, -1, -1, -1, -1};
                    for (std::int64_t i = 0; i < vocab; ++i) {
                        const float value = row[i];
                        if (value > best_v) {
                            best_v = value;
                            best_i = (int)i;
                        }
                        if (value > top_v[4]) {
                            int slot = 4;
                            while (slot > 0 && value > top_v[slot - 1]) {
                                top_v[slot] = top_v[slot - 1];
                                top_i[slot] = top_i[slot - 1];
                                --slot;
                            }
                            top_v[slot] = value;
                            top_i[slot] = (int)i;
                        }
                    }
                    if (best_i == target) ++top1;
                    for (int k = 0; k < 5; ++k) {
                        if (top_i[k] == target) {
                            ++top5;
                            break;
                        }
                    }
                    ++total;
                }
                if (route_hist) {
                    auto& moe_last = *blocks.back().moe;
                    if (moe_last.read_token_counts()) {
                        const auto& picks = moe_last.last_picks();
                        for (std::int64_t r = 0; r < rows_per_chunk && (std::size_t)r < picks.size(); ++r) {
                            const int e = picks[(std::size_t)r];
                            if (e < 0 || e >= kExp) continue;
                            hist[(std::size_t)e][(std::size_t)th[(std::size_t)r]] += 1;
                            expert_tokens[(std::size_t)e] += 1;
                        }
                    }
                }
            }
            const double mean_loss = loss_sum / chunks;
            if (route_hist) {
                std::array<std::vector<int>, 7> top20{};
                for (int e = 0; e < kExp; ++e) {
                    std::vector<int> idx((std::size_t)vocab);
                    std::iota(idx.begin(), idx.end(), 0);
                    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
                                      [&](int a, int b) {
                                          return hist[(std::size_t)e][(std::size_t)a] >
                                                 hist[(std::size_t)e][(std::size_t)b];
                                      });
                    const double denom = (double)std::max<std::int64_t>(expert_tokens[(std::size_t)e], 1);
                    std::printf("ROUTE layer=%d expert=%d tokens=%lld top:",
                                (int)blocks.size() - 1, e, (long long)expert_tokens[(std::size_t)e]);
                    for (int k = 0; k < 5; ++k) {
                        const int token = idx[(std::size_t)k];
                        std::printf(" %d(%.4f)", token,
                                    (double)hist[(std::size_t)e][(std::size_t)token] / denom);
                    }
                    std::printf("\n");
                    top20[(std::size_t)e].assign(idx.begin(), idx.begin() + 20);
                }
                double jacc_sum = 0.0;
                int jacc_n = 0;
                for (int a = 0; a < kExp; ++a) {
                    for (int b = a + 1; b < kExp; ++b) {
                        int inter = 0;
                        for (int x : top20[(std::size_t)a]) {
                            for (int y : top20[(std::size_t)b]) {
                                if (x == y) {
                                    ++inter;
                                    break;
                                }
                            }
                        }
                        jacc_sum += (double)inter / (double)(40 - inter);
                        ++jacc_n;
                    }
                }
                std::printf("ROUTE mean Jaccard(top-20) = %.3f over %d expert pairs\n",
                            jacc_sum / std::max(jacc_n, 1), jacc_n);
            }
            std::printf("EVAL_ONLY steps=%d records=%d tokens=%lld loss=%.4f ppl=%.3f "
                        "top1=%.4f top5=%.4f\n",
                        start_step, chunks * (int)eb, (long long)total, mean_loss,
                        std::exp(mean_loss), (double)top1 / (double)std::max<std::int64_t>(total, 1),
                        (double)top5 / (double)std::max<std::int64_t>(total, 1));
            return 0;
        }

        std::cout << "mn-moe15m pretrain: steps=" << steps_total << " B=" << B << " T=" << T
                  << " dim=" << dim << " layers=" << n_layers << " heads=" << n_head
                  << " lr=" << lr_counter_peak << " lr_fp=" << lr_fp_peak
                  << " experts=7 hidden_exp=" << hidden_exp << " hidden_shared=" << hidden_shared
                  << " balance_gamma=" << balance_gamma << " aux_weight=" << aux_weight
                  << " balance_clamp=" << balance_clamp
                  << " head=" << (tie_head ? "tied" : "untied")
                  << " start=" << start_step << "\n";

        double win_ms = 0.0;
        int win_n = 0;
        double win_gpu_fb_ms = 0.0, win_gpu_opt_ms = 0.0;
        int win_gpu_fb_n = 0, win_gpu_opt_n = 0;
        double loss_ema = -1.0;
        struct StepPhases {
            double upload_ms = 0, record_ms = 0, submit_ms = 0, sync_ms = 0;
            double balance_ms = 0, optim_ms = 0, total_ms = 0, fwd_ms = 0, bwd_record_ms = 0;
        } phases, win_phases;
        auto ms_since = [](std::chrono::steady_clock::time_point a) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        };
            // GPU balance-bias mode: record the per-block sign update inside the optimizer
        // batch instead of a host readback/upload. MOTIFCL_MOE_GPU_BALANCE=0 forces the old
        // host path; unsupported parameters explicitly select it before any batch opens.
        // Default is the CPU path: the same-executable A/B (phase 18) measured +1.0%
        // formal with pair effects +0.1..+2.4%, inside the A/A envelope (0.9055 formal,
        // pairs 0.71..1.08) and with zero CPU-phase delta, so the GPU helper stays
        // opt-in only (MOTIFCL_MOE_GPU_BALANCE=1).
        const bool split_fwd_bwd = env_int("MOTIFCL_MOE_SPLIT_FWD_BWD", 0) != 0;
        const bool gpu_balance_mode = env_int("MOTIFCL_MOE_GPU_BALANCE", 0) != 0 &&
            std::isfinite(balance_gamma) && balance_gamma > 0.0f && std::isfinite(balance_clamp) &&
            balance_clamp > 0.0f && runtime.caps().max_push_constant_bytes >= 48u;
        std::vector<nn::MoE7FFN*> pending_balance;
        pending_balance.reserve(blocks.size());
    for (int step = start_step; step < steps_total; ++step) {
            phases = StepPhases{};
            const auto t0 = std::chrono::steady_clock::now();
            const float lr_c = lr_at(step, lr_counter_peak);
            const float lr_f = lr_at(step, lr_fp_peak);
            set_counter_lr(lr_c);

            Tensor ids, tg;
            upload_batch(train, n_train, B, base_seed * 2654435761u + (unsigned)step, ids, tg, false);
            phases.upload_ms = ms_since(t0);

            for (auto& p : fp) p.t->zero_grad();
            const auto t_rec = std::chrono::steady_clock::now();
            if (!runtime.batch_begin()) { std::cerr << "batch_begin failed\n"; return 1; }
            auto loss = forward_loss(ids, tg, B);
            const double fwd_ms = ms_since(t_rec);
            phases.fwd_ms = fwd_ms;
            double gpu_fb_acc = 0.0;
            bool gpu_fb_seen = false;
            float lv = 0.0f;
            // Load balancing read-back, plus (in split mode) freezing this step's
            // routing for the backward. Both are only legal with no batch open.
            std::array<double, 7> load_sum{};
            int load_n = 0;
            bool load_ok = false;
            auto read_back = [&]() {
                const auto t_sync0 = std::chrono::steady_clock::now();
                lv = loss.item();
                phases.sync_ms = ms_since(t_sync0);
                if (!std::isfinite(lv)) { std::cerr << "non-finite loss at step " << step << "\n"; return false; }
                pending_balance.clear();
                const auto t_bal = std::chrono::steady_clock::now();
                for (auto& blk : blocks) {
                    if (!blk.moe->read_token_counts()) continue;
                    load_ok = true;
                    const auto& load = blk.moe->last_load();
                    for (std::size_t e = 0; e < load.size(); ++e) load_sum[e] += (double)load[e];
                    load_n += 1;
                    blk.moe->prepare_balance_gradient(aux_weight);
                    if (gpu_balance_mode) pending_balance.push_back(blk.moe.get());
                    else blk.moe->update_balance_bias(balance_gamma);
                }
                phases.balance_ms = ms_since(t_bal);
                return true;
            };
            auto submit_batch = [&](const char* what) {
                const auto t_sub = std::chrono::steady_clock::now();
                auto submit = runtime.batch_end();
                phases.submit_ms += ms_since(t_sub);
                if (!submit.success) { std::cerr << what << " batch submit failed: " << submit.error << "\n"; return false; }
                if (gpu_timing) {
                    const double us = runtime.last_gpu_time_us();
                    if (us >= 0.0) {
                        gpu_fb_acc += us * 1e-3;
                        gpu_fb_seen = true;
                    }
                }
                return true;
            };
            // Split mode: submit the forward, read the routing back, then record the
            // backward. Only this ordering lets the backward know which experts won,
            // which is what the zero-gradient expert skip needs. It costs a full
            // forward-length GPU stall while the host records the backward, so it is
            // opt-in: measured neutral with the skip on and a regression without it
            // (see reports/vulkan-moe-sparse-2026-10-21.md, Phase D).
            if (split_fwd_bwd) {
                if (!submit_batch("forward")) return 1;
                if (!read_back()) return 1;
                const auto t_bwd = std::chrono::steady_clock::now();
                if (!runtime.batch_begin()) { std::cerr << "batch_begin (backward) failed\n"; return 1; }
                loss.backward();
                if (!submit_batch("backward")) return 1;
                phases.bwd_record_ms = ms_since(t_bwd);
            } else {
                loss.backward();
                if (!submit_batch("forward+backward")) return 1;
                if (!read_back()) return 1;
            }
            phases.record_ms = ms_since(t_rec);
            const double gpu_fb_ms = gpu_fb_seen ? gpu_fb_acc : -1.0;

            // Auxiliary router gradient + parameter update.
            const auto t_opt = std::chrono::steady_clock::now();
            if (!runtime.batch_begin()) { std::cerr << "batch_begin (optim) failed\n"; return 1; }
            for (auto* moe_ptr : pending_balance) moe_ptr->update_balance_bias(balance_gamma);
            pending_balance.clear();
            for (auto& blk : blocks) blk.moe->inject_balance_gradient();
            for (std::size_t pi = 0; pi < fp.size(); ++pi) {
                auto& p = fp[pi];
                if (!p.t->grad()) { std::cerr << "missing fp grad: param#" << pi << " at step " << step << "\n"; return 1; }
                adam_update_fast(*p.t, *p.t->grad(), p.m, p.v, lr_f, 0.9f, 0.95f, 1e-8f,
                                 step + 1, p.weight_decay);
            }
            double gpu_opt_ms = -1.0;
            {
                auto submit = runtime.batch_end();
                if (!submit.success) { std::cerr << "batch submit failed: " << submit.error << "\n"; return 1; }
                if (gpu_timing) gpu_opt_ms = runtime.last_gpu_time_us() * 1e-3;
            }
            phases.optim_ms = ms_since(t_opt);
            loss_ema = loss_ema < 0 ? lv : 0.99 * loss_ema + 0.01 * lv;

            const auto t1 = std::chrono::steady_clock::now();
            phases.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            win_ms += phases.total_ms;
            win_phases.upload_ms += phases.upload_ms;
            win_phases.fwd_ms += phases.fwd_ms;
            win_phases.record_ms += phases.record_ms;
            win_phases.submit_ms += phases.submit_ms;
            win_phases.sync_ms += phases.sync_ms;
            win_phases.balance_ms += phases.balance_ms;
            win_phases.optim_ms += phases.optim_ms;
            win_phases.bwd_record_ms += phases.bwd_record_ms;
            win_phases.total_ms += phases.total_ms;
            if (std::isfinite(gpu_fb_ms) && gpu_fb_ms >= 0.0) {
                win_gpu_fb_ms += gpu_fb_ms;
                ++win_gpu_fb_n;
            }
            if (std::isfinite(gpu_opt_ms) && gpu_opt_ms >= 0.0) {
                win_gpu_opt_ms += gpu_opt_ms;
                ++win_gpu_opt_n;
            }
            win_n += 1;
            if ((step + 1) % log_every == 0) {
                const double ms = win_ms / win_n;
                const double tok_s = (double)N * 1000.0 / ms;
                const double eta_h = (double)(steps_total - step - 1) * ms / 3.6e6;
                std::printf("step %d loss %.4f ema %.4f lr %.5f  %.0f tok/s  eta %.1f h  devmem %.0f/%.0f MB used %.0f MB",
                            step + 1, lv, loss_ema, lr_c, tok_s, eta_h,
                            (double)runtime.buffer_bytes_live() / 1048576.0,
                            (double)runtime.buffer_bytes_peak() / 1048576.0,
                            (double)runtime.buffer_bytes_requested_peak() / 1048576.0);
                if (gpu_timing) {
                    const double gpu_fb_mean = win_gpu_fb_n == win_n ? win_gpu_fb_ms / win_n : -1.0;
                    const double gpu_opt_mean = win_gpu_opt_n == win_n ? win_gpu_opt_ms / win_n : -1.0;
                    std::printf("  ms fwd+bwd gpu %.2f opt gpu %.2f sync %.2f",
                                gpu_fb_mean, gpu_opt_mean, win_phases.sync_ms / win_n);
                    std::printf(
                        "  phases upload %.1f fwd %.1f record %.1f bwdrec %.1f submit %.1f balance %.1f optim %.1f total %.1f window %d",
                        win_phases.upload_ms / win_n, win_phases.fwd_ms / win_n,
                        win_phases.record_ms / win_n, win_phases.bwd_record_ms / win_n,
                        win_phases.submit_ms / win_n,
                        win_phases.balance_ms / win_n, win_phases.optim_ms / win_n,
                        win_phases.total_ms / win_n, win_n);
                }
                {
                    const char* prof = std::getenv("MOTIFCL_VK_PROFILE");
                    if (prof && *prof && *prof != '0') {
                        const auto rows = runtime.kernel_profile();
                        double total = 0.0;
                        for (const auto& r : rows) total += r.total_us;
                        std::printf("\n  --- kernel GPU profile (window, %zu kernels, %.2f ms) ---",
                                    rows.size(), total * 1e-3);
                        for (const auto& r : rows) {
                            std::printf("\n  %-40s calls %6llu  total %8.2f ms  avg %7.3f ms  %5.1f%%",
                                        r.name.c_str(), (unsigned long long)r.calls, r.total_us * 1e-3,
                                        r.calls ? (r.total_us / (double)r.calls) * 1e-3 : 0.0,
                                        total > 0.0 ? 100.0 * r.total_us / total : 0.0);
                        }
                        std::printf("\n  submit wall (last) %.2f ms  submits %llu", runtime.last_submit_time_us() * 1e-3,
                                    (unsigned long long)runtime.submit_count());
                        runtime.reset_kernel_profile();
                    }
                }
                if (load_ok && load_n > 0) {
                    double lo = 1e9, hi = -1e9;
                    std::printf("  load");
                    for (std::size_t e = 0; e < load_sum.size(); ++e) {
                        const double share = load_sum[e] / load_n;
                        lo = std::min(lo, share);
                        hi = std::max(hi, share);
                        std::printf(" %.3f", share);
                    }
                    std::printf("  spread %.3f", hi - lo);
                    auto bias_host = blocks[0].moe->router_bias.to_vector<float>();
                    double bias_lo = 1e9, bias_hi = -1e9;
                    for (float b : bias_host) {
                        bias_lo = std::min(bias_lo, (double)b);
                        bias_hi = std::max(bias_hi, (double)b);
                    }
                    std::printf("  bias0[%.3f..%.3f]", bias_lo, bias_hi);
                    std::printf("  spread0 %.3f gap0 %.3f",
                                blocks[0].moe->last_logit_spread(),
                                blocks[0].moe->last_logit_gap());
                }
                std::printf("\n");
                std::fflush(stdout);
                win_ms = 0.0;
                win_phases = StepPhases{};
                win_gpu_fb_ms = 0.0;
                win_gpu_opt_ms = 0.0;
                win_gpu_fb_n = 0;
                win_gpu_opt_n = 0;
                win_n = 0;
            }
            if ((step + 1) % eval_every == 0) {
                const float vl = eval_val();
                std::printf("step %d VAL loss %.4f  ppl %.3f\n", step + 1, vl, std::exp((double)vl));
                std::fflush(stdout);
            }
            if (save_ckpt_enabled && ((step + 1) % ckpt_every == 0 || step + 1 == steps_total)) {
                save_ckpt(step + 1);
                std::printf("step %d checkpoint saved\n", step + 1);
                std::fflush(stdout);
            }
        }
        const float vl = eval_val();
        std::printf("FINAL VAL loss %.4f  ppl %.3f\n", vl, std::exp((double)vl));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "train failed: " << e.what() << "\n";
        return 1;
    }
}
