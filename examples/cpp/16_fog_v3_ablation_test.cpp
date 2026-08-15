#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {
std::vector<float> state_code(int state, int n_states, int d_model) {
    std::vector<float> out(static_cast<std::size_t>(d_model));
    const int pairs = d_model / 2;
    const float amp = std::sqrt(2.0f);
    for (int j = 0; j < pairs; ++j) {
        const int h = j + 1;
        const float angle = 2.0f * 3.14159265358979323846f * static_cast<float>(state * h) / static_cast<float>(n_states);
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

int decode_state(const motifcl::Tensor& value, const motifcl::Tensor& codebook, const motifcl::Tensor& norm_weight, int n_states) {
    auto normed = motifcl::rmsnorm(value, norm_weight);
    auto logits = motifcl::matmul_transpose_b(normed, codebook);
    auto lhost = logits.to_vector<float>();
    int best = 0;
    float best_v = -1e9f;
    for (int s = 0; s < n_states; ++s) {
        const int tok = 4 + s;
        if (lhost[static_cast<std::size_t>(tok)] > best_v) {
            best_v = lhost[static_cast<std::size_t>(tok)];
            best = s;
        }
    }
    return best;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string checkpoint = "checkpoints/fog_v3_rx580_reasoning_sft.mclp";
        if (argc > 1) checkpoint = argv[1];

        auto backend = motifcl::Backend::create_vulkan();
        const int n_states = 16;
        motifcl::nn::FogV3Config cfg;
        cfg.vocab_size = 8192;
        cfg.max_seq_len = 512;
        cfg.d_model = 320;
        cfg.n_heads = 5;
        cfg.n_layers = 4;
        cfg.d_ff = 1344;
        cfg.dropout = 0.0f;
        motifcl::nn::FogV3Model model(backend, cfg);

        auto params = model.parameters();
        if (!std::filesystem::exists(checkpoint)) {
            std::cerr << "Checkpoint not found: " << checkpoint << "\n";
            return 1;
        }
        motifcl::load_parameters(params, backend, checkpoint);
        install_state_codebook(model, n_states);
        motifcl::autograd::NoGradGuard no_grad;

        auto norm_weight = motifcl::Tensor::ones(backend, {cfg.d_model}, motifcl::DType::F32);
        norm_weight.set_requires_grad(false);
        auto codebook = motifcl::rmsnorm(model.lexical.token_embedding.weight.data, norm_weight);
        codebook.set_requires_grad(false);

        const int num_tests = 50;
        std::vector<int> depths = {1, 2, 3, 4, 6, 8, 10, 12};

        std::cout << "========================================================================\n";
        std::cout << "  СРАВНИТЕЛЬНЫЙ ТЕСТ: ЧИСТАЯ LLM ПРОТИВ ЛОГИЧЕСКОГО ДВИЖКА FOG v3\n";
        std::cout << "  (Запуск на 50 случайных независимых задачах на каждую глубину)\n";
        std::cout << "========================================================================\n";
        std::cout << " Глубина (Шаги R) | Чистая LLM (4 слоя) | Логический движок FOG | Разница\n";
        std::cout << "------------------+---------------------+-----------------------+---------\n";

        std::mt19937 rng(98765u);
        std::uniform_int_distribution<int> sdist(0, n_states - 1);

        for (int depth : depths) {
            int fog_correct = 0;
            int pure_llm_correct = 0;

            for (int trial = 0; trial < num_tests; ++trial) {
                // Generate a random problem
                int start_state = sdist(rng);
                std::vector<int> table(static_cast<std::size_t>(n_states));
                for (int s = 0; s < n_states; ++s) table[static_cast<std::size_t>(s)] = sdist(rng);

                int expected = start_state;
                for (int t = 0; t < depth; ++t) {
                    expected = (expected + table[static_cast<std::size_t>(expected)]) % n_states;
                }

                // 1. Evaluate with FOG Register Engine
                std::int32_t q_tok = 4 + start_state;
                auto q_tensor = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &q_tok);
                auto reg_state = model.initial_state(q_tensor);

                std::vector<std::int32_t> k_ids(static_cast<std::size_t>(n_states));
                std::vector<std::int32_t> v_ids(static_cast<std::size_t>(n_states));
                for (int s = 0; s < n_states; ++s) {
                    k_ids[static_cast<std::size_t>(s)] = 4 + s;
                    v_ids[static_cast<std::size_t>(s)] = 4 + table[static_cast<std::size_t>(s)];
                }
                auto k_tensor = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, k_ids.data());
                auto v_tensor = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, v_ids.data());

                for (int t = 0; t < depth; ++t) {
                    auto out = model.structured_step(reg_state, k_tensor, v_tensor, 1, n_states);
                    reg_state = out.state;
                }
                const int fog_decoded = decode_state(reg_state.value, codebook, norm_weight, n_states);
                if (fog_decoded == expected) ++fog_correct;

                // 2. Evaluate with Pure Lexical Backbone (Direct token prediction through 4 Transformer layers)
                std::vector<std::int32_t> prompt_tokens;
                prompt_tokens.push_back(4 + start_state);
                for (int s = 0; s < n_states; ++s) {
                    prompt_tokens.push_back(4 + table[static_cast<std::size_t>(s)]);
                }
                auto tokens_t = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(prompt_tokens.size())}, motifcl::DType::I32, prompt_tokens.data());
                auto logits3 = model.forward(tokens_t);
                auto lhost = logits3.to_vector<float>();
                const int last_pos = static_cast<int>(prompt_tokens.size()) - 1;
                const float* last_l = lhost.data() + last_pos * cfg.vocab_size;
                int llm_best = 0;
                float llm_best_v = -1e9f;
                for (int s = 0; s < n_states; ++s) {
                    if (last_l[4 + s] > llm_best_v) {
                        llm_best_v = last_l[4 + s];
                        llm_best = s;
                    }
                }
                if (llm_best == expected) ++pure_llm_correct;
            }

            const float fog_acc = static_cast<float>(fog_correct) * 100.0f / static_cast<float>(num_tests);
            const float llm_acc = static_cast<float>(pure_llm_correct) * 100.0f / static_cast<float>(num_tests);

            std::cout << "      R = " << std::setw(2) << depth
                      << "      |        " << std::setw(5) << std::fixed << std::setprecision(1) << llm_acc << "%       "
                      << "|        " << std::setw(5) << std::fixed << std::setprecision(1) << fog_acc << "%        "
                      << "|  +" << std::setw(5) << std::fixed << std::setprecision(1) << (fog_acc - llm_acc) << "%\n";
        }
        std::cout << "========================================================================\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
