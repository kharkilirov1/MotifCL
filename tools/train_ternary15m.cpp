// ternary15M-from-scratch pretrain runner on Vulkan (RX 580 class).
//
// Faithful twin of the stories15M/ternary15M setup (dim 288, 6 layers, MHA-6,
// SwiGLU 768, vocab 32k, seq 256, RoPE theta 1e4, RMSNorm, tied embeddings)
// with ONE deliberate difference: the 42 attention/FFN linears are
// CounterStateLinear ternary counter synapses — there is NO FP32 latent weight
// and no Adam state for them at all (the reference trains FP32 latents with
// STE + AdamW and discards them; here the counter state IS the model).
// FP32 trainables: tied token embedding (AdamW) + RMSNorm gains (Adam).
//
// Data: packed uint16 records of length seq+1 produced by the ternary15M
// preprocess.py (BOS + sliding window, little-endian), train.bin/val.bin.
//
// env: DATA_DIR (required), OUT_DIR (required), STEPS, BATCH, SEQ, LR
// (counter peak), LR_FP (embedding/norm peak), WARMUP_FRAC, MIN_LR_FRAC,
// EVAL_EVERY, EVAL_RECORDS, CKPT_EVERY, LOG_EVERY, SEED, RESUME (default 1).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/embedding.hpp>
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

        const std::string data_dir = env_str("DATA_DIR", "");
        const std::string out_dir = env_str("OUT_DIR", "");
        if (data_dir.empty() || out_dir.empty()) {
            std::cerr << "DATA_DIR and OUT_DIR are required\n";
            return 1;
        }
        std::filesystem::create_directories(out_dir);

        const int steps_total = env_int("STEPS", 160000);
        const std::int64_t B = env_int("BATCH", 16);
        const std::int64_t T = env_int("SEQ", 256);
        const float lr_counter_peak = env_float("LR", 4e-3f);
        const float lr_fp_peak = env_float("LR_FP", 6e-3f);
        const float warmup_frac = env_float("WARMUP_FRAC", 0.02f);
        const float min_lr_frac = env_float("MIN_LR_FRAC", 0.1f);
        const int eval_every = env_int("EVAL_EVERY", 500);
        const int eval_records = env_int("EVAL_RECORDS", 96);
        const int ckpt_every = env_int("CKPT_EVERY", 1000);
        const int log_every = env_int("LOG_EVERY", 20);
        const unsigned base_seed = (unsigned)env_int("SEED", 1337);
        const bool resume = env_int("RESUME", 1) != 0;

        const std::int64_t dim = 288, hidden = 768, vocab = 32000;
        const int n_head = 6, n_kv_head = 6, n_layers = 6, C = 11;
        const std::int64_t N = B * T;
        const float eps = 1e-5f;
        const std::int64_t rec_len = T + 1;

        // ---- resume bookkeeping (must be known before layer construction:
        // counter seeds advance once per update, so seed = base + start_step
        // reproduces the RNG stream of an uninterrupted run) ----
        int start_step = 0;
        const std::string meta_path = out_dir + "/meta.txt";
        if (resume && std::filesystem::exists(meta_path)) {
            std::ifstream mf(meta_path);
            mf >> start_step;
            std::cout << "resuming from step " << start_step << "\n";
        }

        // ---- data ----
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
        // Untied LM head: the Vulkan matmul_transpose_b path has no autograd
        // node yet, so a tied head silently severs the whole backward chain.
        // The 42 ternary linears are unaffected; revisit when tied autograd lands.
        std::vector<float> head_host((std::size_t)(dim * vocab));
        {
            std::mt19937 rng(base_seed + 7u);
            std::normal_distribution<float> gauss(0.0f, 0.02f);
            for (auto& x : head_host) x = gauss(rng);
        }
        auto head = Tensor::from_cpu(backend, {dim, vocab}, DType::F32, head_host.data());
        head.set_requires_grad(true);
        const std::int64_t max_flat = std::max(N, (std::int64_t)32 * T);
        std::vector<float> pos_zero((std::size_t)(max_flat * dim), 0.0f);
        auto pos_w = Tensor::from_cpu(backend, {max_flat, dim}, DType::F32, pos_zero.data());

        struct Block {
            std::unique_ptr<nn::CounterStateLinear> wq, wk, wv, wo, wup, wdown;
            Tensor norm1, norm2;
        };
        std::vector<float> ones((std::size_t)dim, 1.0f);
        std::vector<Block> blocks;
        const float counter_init_lr = lr_counter_peak;
        for (int l = 0; l < n_layers; ++l) {
            const unsigned s0 = 1000u * (unsigned)(l + 1) + (unsigned)start_step;
            Block blk{
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, counter_init_lr, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 1),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, counter_init_lr, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 2),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, counter_init_lr, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 3),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)dim, C, counter_init_lr, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 4),
                std::make_unique<nn::CounterStateLinear>(backend, (int)dim, (int)(hidden * 2), C, counter_init_lr, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 5),
                std::make_unique<nn::CounterStateLinear>(backend, (int)hidden, (int)dim, C, counter_init_lr, 0.0f, 1.0f, 0.9f, 1e-3f, s0 + 6),
                Tensor::from_cpu(backend, {dim}, DType::F32, ones.data()),
                Tensor::from_cpu(backend, {dim}, DType::F32, ones.data()),
            };
            blk.norm1.set_requires_grad(true);
            blk.norm2.set_requires_grad(true);
            blocks.push_back(std::move(blk));
        }
        auto norm_f = Tensor::from_cpu(backend, {dim}, DType::F32, ones.data());
        norm_f.set_requires_grad(true);

        // ---- fp optimizer state (Adam; wd only on the tied embedding, like
        // the reference's decay/no-decay split) ----
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
        add_fp(&head, 0.1f);
        for (auto& blk : blocks) { add_fp(&blk.norm1, 0.0f); add_fp(&blk.norm2, 0.0f); }
        add_fp(&norm_f, 0.0f);

        // ---- checkpoint io ----
        auto ckpt_path = [&](const std::string& name) { return out_dir + "/" + name; };
        auto save_ckpt = [&](int step) {
            save_tensor(emb, ckpt_path("emb.mt"));
            save_tensor(head, ckpt_path("head.mt"));
            save_tensor(norm_f, ckpt_path("norm_f.mt"));
            for (int l = 0; l < n_layers; ++l) {
                auto& blk = blocks[(std::size_t)l];
                const std::string p = "l" + std::to_string(l) + "_";
                save_tensor(blk.norm1, ckpt_path(p + "n1.mt"));
                save_tensor(blk.norm2, ckpt_path(p + "n2.mt"));
                nn::CounterStateLinear* ls[6] = {blk.wq.get(), blk.wk.get(), blk.wv.get(),
                                                 blk.wo.get(), blk.wup.get(), blk.wdown.get()};
                const char* nm[6] = {"wq", "wk", "wv", "wo", "wup", "wdown"};
                for (int i = 0; i < 6; ++i) {
                    save_tensor(ls[i]->state, ckpt_path(p + nm[i] + "_state.mt"));
                    save_tensor(ls[i]->scale, ckpt_path(p + nm[i] + "_scale.mt"));
                    save_tensor(ls[i]->v, ckpt_path(p + nm[i] + "_v.mt"));
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
            load_into(head, ckpt_path("head.mt"));
            head.set_requires_grad(true);
            load_into(norm_f, ckpt_path("norm_f.mt"));
            norm_f.set_requires_grad(true);
            for (int l = 0; l < n_layers; ++l) {
                auto& blk = blocks[(std::size_t)l];
                const std::string p = "l" + std::to_string(l) + "_";
                load_into(blk.norm1, ckpt_path(p + "n1.mt"));
                blk.norm1.set_requires_grad(true);
                load_into(blk.norm2, ckpt_path(p + "n2.mt"));
                blk.norm2.set_requires_grad(true);
                nn::CounterStateLinear* ls[6] = {blk.wq.get(), blk.wk.get(), blk.wv.get(),
                                                 blk.wo.get(), blk.wup.get(), blk.wdown.get()};
                const char* nm[6] = {"wq", "wk", "wv", "wo", "wup", "wdown"};
                for (int i = 0; i < 6; ++i) {
                    ls[i]->state = load_tensor(backend, ckpt_path(p + nm[i] + "_state.mt"));
                    ls[i]->scale = load_tensor(backend, ckpt_path(p + nm[i] + "_scale.mt"));
                    ls[i]->v = load_tensor(backend, ckpt_path(p + nm[i] + "_v.mt"));
                }
            }
            for (std::size_t i = 0; i < fp.size(); ++i) {
                fp[i].m = load_tensor(backend, ckpt_path("adam_m_" + std::to_string(i) + ".mt"));
                fp[i].v = load_tensor(backend, ckpt_path("adam_v_" + std::to_string(i) + ".mt"));
            }
            // fp param tensor pointers were replaced by load_into; rebind.
            fp[0].t = &emb;
            fp[1].t = &head;
            std::size_t k = 2;
            for (auto& blk : blocks) { fp[k++].t = &blk.norm1; fp[k++].t = &blk.norm2; }
            fp[k].t = &norm_f;
        };
        if (start_step > 0) load_ckpt();

        // ---- forward for one uploaded batch ----
        auto forward_loss = [&](const Tensor& ids, const Tensor& targets, std::int64_t bs) {
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
                auto m = swiglu(blk.wup->forward(b2));
                cur = add(h, blk.wdown->forward(m));
            }
            auto xf = rmsnorm(cur, norm_f, eps);
            auto logits = matmul(xf, head);   // untied (see note above)
            return softmax_cross_entropy(logits, targets);
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
            for (auto& blk : blocks)
                for (auto* l : {blk.wq.get(), blk.wk.get(), blk.wv.get(), blk.wo.get(),
                                blk.wup.get(), blk.wdown.get()})
                    l->set_lr(lr);
        };
        auto set_counter_training = [&](bool t) {
            for (auto& blk : blocks)
                for (auto* l : {blk.wq.get(), blk.wk.get(), blk.wv.get(), blk.wo.get(),
                                blk.wup.get(), blk.wdown.get()})
                    l->set_training(t);
        };

        auto eval_val = [&]() {
            set_counter_training(false);
            const std::int64_t eb = 32;
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

        std::cout << "ternary15m pretrain: steps=" << steps_total << " B=" << B << " T=" << T
                  << " lr=" << lr_counter_peak << " lr_fp=" << lr_fp_peak
                  << " start=" << start_step << "\n";

        double win_ms = 0.0;
        int win_n = 0;
        double loss_ema = -1.0;
        for (int step = start_step; step < steps_total; ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            const float lr_c = lr_at(step, lr_counter_peak);
            const float lr_f = lr_at(step, lr_fp_peak);
            set_counter_lr(lr_c);

            Tensor ids, tg;
            upload_batch(train, n_train, B, base_seed * 2654435761u + (unsigned)step, ids, tg, false);

            for (auto& p : fp) p.t->zero_grad();
            if (!runtime.batch_begin()) { std::cerr << "batch_begin failed\n"; return 1; }
            auto loss = forward_loss(ids, tg, B);
            loss.backward();
            for (std::size_t pi = 0; pi < fp.size(); ++pi) {
                auto& p = fp[pi];
                if (!p.t->grad()) { std::cerr << "missing fp grad: param#" << pi << " at step " << step << "\n"; return 1; }
                adam_update_fast(*p.t, *p.t->grad(), p.m, p.v, lr_f, 0.9f, 0.95f, 1e-8f,
                                 step + 1, p.weight_decay);
            }
            {
                auto submit = runtime.batch_end();
                if (!submit.success) { std::cerr << "batch submit failed: " << submit.error << "\n"; return 1; }
            }
            const float lv = loss.item();
            if (!std::isfinite(lv)) { std::cerr << "non-finite loss at step " << step << "\n"; return 1; }
            loss_ema = loss_ema < 0 ? lv : 0.99 * loss_ema + 0.01 * lv;

            const auto t1 = std::chrono::steady_clock::now();
            win_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            win_n += 1;
            if ((step + 1) % log_every == 0) {
                const double ms = win_ms / win_n;
                const double tok_s = (double)N * 1000.0 / ms;
                const double eta_h = (double)(steps_total - step - 1) * ms / 3.6e6;
                std::printf("step %d loss %.4f ema %.4f lr %.5f  %.0f tok/s  eta %.1f h\n",
                            step + 1, lv, loss_ema, lr_c, tok_s, eta_h);
                std::fflush(stdout);
                win_ms = 0.0; win_n = 0;
            }
            if ((step + 1) % eval_every == 0) {
                const float vl = eval_val();
                std::printf("step %d VAL loss %.4f\n", step + 1, vl);
                std::fflush(stdout);
            }
            if ((step + 1) % ckpt_every == 0 || step + 1 == steps_total) {
                save_ckpt(step + 1);
                std::printf("step %d checkpoint saved\n", step + 1);
                std::fflush(stdout);
            }
        }
        const float vl = eval_val();
        std::printf("FINAL VAL loss %.4f\n", vl);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "train failed: " << e.what() << "\n";
        return 1;
    }
}
