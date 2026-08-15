#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

namespace {
constexpr float kPi = 3.14159265358979323846f;

int arg_int(char** argv, int argc, int idx, int fallback) {
    if (idx >= argc) return fallback;
    return std::max(1, std::atoi(argv[idx]));
}
float arg_float(char** argv, int argc, int idx, float fallback) {
    if (idx >= argc) return fallback;
    return std::max(1e-8f, std::strtof(argv[idx], nullptr));
}

std::size_t parameter_count(const std::vector<motifcl::nn::Parameter*>& params) {
    std::size_t n = 0;
    for (auto* p : params) if (p && p->data.valid()) n += static_cast<std::size_t>(p->data.numel());
    return n;
}

std::vector<float> state_code(int state, int n_states, int d_model) {
    std::vector<float> out(static_cast<std::size_t>(d_model));
    const int pairs = d_model / 2;
    const float amp = std::sqrt(2.0f);
    for (int j = 0; j < pairs; ++j) {
        const int h = j + 1;
        const float angle = 2.0f * kPi * static_cast<float>(state * h) / static_cast<float>(n_states);
        out[static_cast<std::size_t>(2 * j)] = amp * std::cos(angle);
        out[static_cast<std::size_t>(2 * j + 1)] = amp * std::sin(angle);
    }
    return out;
}

void install_state_codebook(motifcl::nn::FogV3Model& model, int n_states) {
    auto host = model.lexical.token_embedding.weight.data.to_vector<float>();
    const int vocab = model.config.vocab_size;
    const int d = model.config.d_model;
    if (vocab < 4 + n_states) throw std::runtime_error("vocab too small for state codebook");
    for (int s = 0; s < n_states; ++s) {
        const auto code = state_code(s, n_states, d);
        std::copy(code.begin(), code.end(), host.begin() + static_cast<std::ptrdiff_t>((4 + s) * d));
    }
    auto& backend = model.lexical.token_embedding.weight.data.backend();
    model.lexical.token_embedding.weight.data = motifcl::Tensor::from_cpu(backend, {vocab, d}, motifcl::DType::F32, host.data());
    model.lexical.token_embedding.weight.data.set_requires_grad(false);
    model.lexical.token_embedding.weight.trainable = false;
}

struct Batch {
    std::vector<std::int32_t> query;
    std::vector<std::int32_t> keys;
    std::vector<std::int32_t> values;
    std::vector<std::int32_t> target;
};

Batch make_reasoning_batch(int n_states, int batch, int depth, std::mt19937& rng) {
    std::uniform_int_distribution<int> state_dist(0, n_states - 1);
    Batch out;
    out.query.resize(static_cast<std::size_t>(batch));
    out.keys.resize(static_cast<std::size_t>(batch * n_states));
    out.values.resize(static_cast<std::size_t>(batch * n_states));
    out.target.resize(static_cast<std::size_t>(batch));
    std::vector<int> table(static_cast<std::size_t>(batch * n_states));
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < n_states; ++s) {
            const int operand = state_dist(rng);
            table[static_cast<std::size_t>(b * n_states + s)] = operand;
            out.keys[static_cast<std::size_t>(b * n_states + s)] = 4 + s;
            out.values[static_cast<std::size_t>(b * n_states + s)] = 4 + operand;
        }
        int cur = state_dist(rng);
        out.query[static_cast<std::size_t>(b)] = 4 + cur;
        for (int t = 0; t < depth; ++t) {
            const int operand = table[static_cast<std::size_t>(b * n_states + cur)];
            cur = (cur + operand) % n_states;
        }
        out.target[static_cast<std::size_t>(b)] = 4 + cur;
    }
    return out;
}

int argmax_row(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return best;
}

struct Eval { float accuracy = 0.f; float block_fraction = 0.f; };

Eval evaluate(motifcl::nn::FogV3Model& model, motifcl::Backend& backend, const motifcl::Tensor& codebook, const motifcl::Tensor& norm_weight, int n_states, int depth, int examples, std::uint32_t seed) {
    motifcl::autograd::NoGradGuard no_grad;
    std::mt19937 rng(seed);
    int correct = 0, total = 0, block = 0, routes = 0;
    const int vocab = model.config.vocab_size;
    const int d_model = model.config.d_model;
    auto& runtime = backend.vulkan_runtime();
    while (total < examples) {
        const int bs = std::min(64, examples - total);
        auto bh = make_reasoning_batch(n_states, bs, depth, rng);
        auto q = motifcl::Tensor::from_cpu(backend, {bs}, motifcl::DType::I32, bh.query.data());
        auto k = motifcl::Tensor::from_cpu(backend, {bs * n_states}, motifcl::DType::I32, bh.keys.data());
        auto v = motifcl::Tensor::from_cpu(backend, {bs * n_states}, motifcl::DType::I32, bh.values.data());

        const bool batched = runtime.batch_begin();
        auto state = model.initial_state(q);
        motifcl::Tensor last_route;
        for (int t = 0; t < depth; ++t) {
            auto step_out = model.structured_step(state, k, v, bs, n_states);
            state = step_out.state;
            last_route = step_out.operator_logits;
        }
        auto s_norm = motifcl::rmsnorm(state.value, norm_weight);
        auto logits = motifcl::scale(motifcl::matmul_transpose_b(s_norm, codebook), 20.0f / static_cast<float>(d_model));
        if (batched) {
            const auto submit = runtime.batch_end();
            if (!submit.success) throw std::runtime_error("Eval batch submit failed: " + submit.error);
        }
        backend.finish();

        if (last_route.valid()) {
            auto rhost = last_route.to_vector<float>();
            for (int r = 0; r < bs; ++r) {
                if (argmax_row(rhost.data() + r * 7, 7) == 2) ++block;
                ++routes;
            }
        }

        auto lhost = logits.to_vector<float>();
        for (int b = 0; b < bs; ++b) {
            const int pred = argmax_row(lhost.data() + b * vocab, vocab);
            if (pred == bh.target[static_cast<std::size_t>(b)]) ++correct;
            ++total;
        }
    }
    return {static_cast<float>(correct) / static_cast<float>(total),
            routes ? static_cast<float>(block) / static_cast<float>(routes) : 0.f};
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string pretrained_weights = argc > 1 ? argv[1] : "checkpoints/fog_v3_rx580_lexical_30k.mclp";
        const std::string output_checkpoint = argc > 2 ? argv[2] : "checkpoints/fog_v3_rx580_reasoning_sft.mclp";
        const int n_states = arg_int(argv, argc, 3, 16);
        const int batch_size = arg_int(argv, argc, 4, 32);
        const float base_lr = arg_float(argv, argc, 5, 2e-3f);

        auto backend = motifcl::Backend::create_vulkan();
        const auto info = backend.device_info();

        motifcl::nn::FogV3Config cfg;
        cfg.vocab_size = 8192;
        cfg.max_seq_len = 512;
        cfg.d_model = 320;
        cfg.n_heads = 5;
        cfg.n_layers = 4;
        cfg.d_ff = 1344;
        cfg.dropout = 0.0f;
        motifcl::nn::FogV3Model model(backend, cfg);

        auto all_params = model.parameters();
        if (std::filesystem::exists(pretrained_weights)) {
            motifcl::load_parameters(all_params, backend, pretrained_weights);
            std::cout << "Loaded pretrained lexical backbone: " << pretrained_weights << "\n";
        }
        install_state_codebook(model, n_states);

        auto norm_weight = motifcl::Tensor::ones(backend, {cfg.d_model});
        norm_weight.set_requires_grad(false);
        auto codebook = motifcl::rmsnorm(model.lexical.token_embedding.weight.data, norm_weight);
        codebook.set_requires_grad(false);

        auto compute_logits = [&](const motifcl::Tensor& val) {
            auto state = motifcl::rmsnorm(val, norm_weight);
            return motifcl::scale(motifcl::matmul_transpose_b(state, codebook), 20.0f / static_cast<float>(cfg.d_model));
        };

        auto train_params = model.machine_parameters();
        std::cout << "FOG v3 Reasoning SFT (Latent Register Machine)\n"
                  << "Device: " << info.device_name << "\n"
                  << "Machine params: " << parameter_count(train_params)
                  << " | Total params: " << parameter_count(all_params) << "\n"
                  << "Batch: " << batch_size << " | States: " << n_states << " | Base LR: " << base_lr << "\n\n";

        struct Stage {
            std::string name;
            int depth;
            int steps;
        };
        const std::vector<Stage> curriculum = {
            {"Stage A (R=1 Semantic Warmup)", 1, 100},
            {"Stage B (R=2 Composition)",    2, 100},
            {"Stage C (R=4 Deep Recurrence)", 4, 150}
        };

        motifcl::optim::Adam opt(train_params, base_lr, 0.9f, 0.95f, 1e-8f, 0.0f);
        std::mt19937 rng(20260815u);
        auto& runtime = backend.vulkan_runtime();

        for (const auto& stage : curriculum) {
            std::cout << "=== Starting " << stage.name << " (Depth R=" << stage.depth << ", Steps=" << stage.steps << ") ===\n";
            for (int step = 1; step <= stage.steps; ++step) {
                const float progress = static_cast<float>(step) / static_cast<float>(stage.steps);
                const float cur_lr = base_lr * (0.2f + 0.8f * 0.5f * (1.0f + std::cos(kPi * progress)));
                opt.set_lr(cur_lr);

                auto bh = make_reasoning_batch(n_states, batch_size, stage.depth, rng);
                auto q = motifcl::Tensor::from_cpu(backend, {batch_size}, motifcl::DType::I32, bh.query.data());
                auto k = motifcl::Tensor::from_cpu(backend, {batch_size * n_states}, motifcl::DType::I32, bh.keys.data());
                auto v = motifcl::Tensor::from_cpu(backend, {batch_size * n_states}, motifcl::DType::I32, bh.values.data());
                auto target = motifcl::Tensor::from_cpu(backend, {batch_size}, motifcl::DType::I32, bh.target.data());

                const auto t0 = std::chrono::steady_clock::now();
                const bool batched = runtime.batch_begin();
                auto state = model.initial_state(q);
                for (int t = 0; t < stage.depth; ++t) {
                    state = model.structured_step(state, k, v, batch_size, n_states).state;
                }
                auto logits = compute_logits(state.value);
                auto loss = motifcl::softmax_cross_entropy(logits, target);
                loss.backward();
                opt.step();
                opt.zero_grad();
                if (batched) {
                    const auto submit = runtime.batch_end();
                    if (!submit.success) throw std::runtime_error("Batch submit failed: " + submit.error);
                }
                backend.finish();
                const auto t1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                if (step == 1 || step % 25 == 0 || step == stage.steps) {
                    const float lv = loss.item();
                    std::cout << "  step=" << std::setw(3) << step << "/" << stage.steps
                              << " loss=" << std::scientific << std::setprecision(4) << lv << std::defaultfloat
                              << " lr=" << std::scientific << std::setprecision(2) << cur_lr << std::defaultfloat
                              << " time=" << std::fixed << std::setprecision(1) << ms << "ms" << std::endl;
                }
            }
            std::cout << std::endl;
        }

        std::cout << "=== Final Generalization Evaluation across Recurrent Depths ===\n" << std::flush;
        for (int eval_depth : {1, 2, 3, 4, 6, 8}) {
            const auto ev = evaluate(model, backend, codebook, norm_weight, n_states, eval_depth, 128, 999u + static_cast<std::uint32_t>(eval_depth));
            std::cout << "Depth R=" << std::setw(2) << eval_depth
                      << " | Accuracy: " << std::fixed << std::setprecision(1) << (ev.accuracy * 100.0f) << "%"
                      << " | BlockProduct Route: " << std::setprecision(1) << (ev.block_fraction * 100.0f) << "%" << std::endl;
        }

        std::filesystem::create_directories(std::filesystem::path(output_checkpoint).parent_path());
        motifcl::save_parameters(all_params, output_checkpoint);
        std::cout << "\nSaved reasoning checkpoint: " << output_checkpoint << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
