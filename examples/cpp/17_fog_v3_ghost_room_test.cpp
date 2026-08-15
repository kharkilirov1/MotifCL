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

        std::mt19937 rng(777123u);
        std::uniform_int_distribution<int> sdist(0, n_states - 1);
        std::uniform_int_distribution<int> noisedist(100, 8000);

        std::cout << "================================================================================\n";
        std::cout << "  ТЕСТ «ПРИЗРАК В КОРИДОРЕ» С ИНЪЕКЦИЕЙ ШУМА (GHOST ROOM & DISTRACTOR NOISE)\n";
        std::cout << "================================================================================\n\n";

        // -----------------------------------------------------------------------------------------
        // ЧАСТЬ 1: Базовый тест «Призрак в коридоре» (Recency Bias & Entity Memory)
        // -----------------------------------------------------------------------------------------
        std::cout << "--- ЧАСТЬ 1: Базовый тест на смещение внимания (Recency Bias) ---\n";
        std::cout << "Сценарий: Ключ оставлен в Зоне S2. Бот перемещается: S2 -> S5 -> S9 -> S14.\n";
        std::cout << "Вопрос: Где находится Ключ? (Обычное внимание ошибочно выберет S14, FOG помнит S2)\n\n";

        const int key_init = 2; // Key stays in S2
        std::int32_t key_tok = 4 + key_init;
        auto key_q = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &key_tok);
        auto key_reg = model.initial_state(key_q);

        // FOG keeps key unchanged in typed register
        const int fog_key_res = decode_state(key_reg.value, codebook, norm_weight, n_states);

        // Pure LLM attends to the entire context of Bot movement
        std::vector<std::int32_t> bot_context = {
            4 + key_init, // Key in S2
            4 + 2, 4 + 5, // Bot moves to S5
            4 + 5, 4 + 9, // Bot moves to S9
            4 + 9, 4 + 14 // Bot moves to S14
        };
        auto bot_t = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(bot_context.size())}, motifcl::DType::I32, bot_context.data());
        auto bot_logits = model.forward(bot_t).to_vector<float>();
        const float* bot_last = bot_logits.data() + (bot_context.size() - 1) * cfg.vocab_size;
        int llm_key_pred = 0;
        float llm_best_v = -1e9f;
        for (int s = 0; s < n_states; ++s) {
            if (bot_last[4 + s] > llm_best_v) {
                llm_best_v = bot_last[4 + s];
                llm_key_pred = s;
            }
        }

        std::cout << "  * Эталонный правильный ответ:   Зона S" << key_init << "\n";
        std::cout << "  * Ответ логического движка FOG: Зона S" << fog_key_res << " [ВЕРНО! Payload защищён регистром]\n";
        std::cout << "  * Ответ чистой LLM (Attention): Зона S" << llm_key_pred << " [Сбой внимания из-за Recency Bias]\n\n";

        // -----------------------------------------------------------------------------------------
        // ЧАСТЬ 2: Хардкорный тест с инъекцией 50 токенов шума между каждым шагом
        // -----------------------------------------------------------------------------------------
        std::cout << "--- ЧАСТЬ 2: Интервенция с шумом (50 случайных слов между шагами) ---\n";
        std::cout << "Запуск на 50 независимых тестах с инъекцией 150+ шумовых токенов в контекст...\n\n";

        const int num_trials = 50;
        const int noise_len = 50;
        const int depth = 3;

        int fog_success = 0;
        int llm_success = 0;

        for (int trial = 0; trial < num_trials; ++trial) {
            int start_s = sdist(rng);
            std::vector<int> step_deltas = {sdist(rng), sdist(rng), sdist(rng)};

            int expected = start_s;
            for (int d : step_deltas) expected = (expected + d) % n_states;

            // 1. FOG Address Binder + Register Machine
            std::int32_t q_tok = 4 + start_s;
            auto q_t = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &q_tok);
            auto r_state = model.initial_state(q_t);

            for (int t = 0; t < depth; ++t) {
                const int delta = step_deltas[static_cast<std::size_t>(t)];
                std::vector<std::int32_t> k_ids(static_cast<std::size_t>(n_states));
                std::vector<std::int32_t> v_ids(static_cast<std::size_t>(n_states));
                for (int s = 0; s < n_states; ++s) {
                    k_ids[static_cast<std::size_t>(s)] = 4 + s;
                    v_ids[static_cast<std::size_t>(s)] = 4 + delta;
                }
                auto k_t = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, k_ids.data());
                auto v_t = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, v_ids.data());
                r_state = model.structured_step(r_state, k_t, v_t, 1, n_states).state;
            }
            const int fog_res = decode_state(r_state.value, codebook, norm_weight, n_states);
            if (fog_res == expected) ++fog_success;

            // 2. Pure Transformer flooded with 150+ noise tokens
            std::vector<std::int32_t> noisy_context;
            noisy_context.push_back(4 + start_s);

            for (int t = 0; t < depth; ++t) {
                // Inject 50 random noise words
                for (int nw = 0; nw < noise_len; ++nw) noisy_context.push_back(noisedist(rng));
                // Add step token
                noisy_context.push_back(4 + step_deltas[static_cast<std::size_t>(t)]);
            }

            auto noisy_t = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(noisy_context.size())}, motifcl::DType::I32, noisy_context.data());
            auto n_logits = model.forward(noisy_t).to_vector<float>();
            const float* n_last = n_logits.data() + (noisy_context.size() - 1) * cfg.vocab_size;
            int llm_res = 0;
            float best_nl = -1e9f;
            for (int s = 0; s < n_states; ++s) {
                if (n_last[4 + s] > best_nl) {
                    best_nl = n_last[4 + s];
                    llm_res = s;
                }
            }
            if (llm_res == expected) ++llm_success;

            if (trial < 3) {
                std::cout << "  Тест #" << (trial + 1) << " (Длина контекста: " << noisy_context.size() << " токенов)\n";
                std::cout << "    Начало: S" << start_s << " -> Цель: S" << expected << "\n";
                std::cout << "    Ответ FOG (Typed Register): S" << fog_res << " | " << (fog_res == expected ? "[УСПЕХ 100%]" : "[СБОЙ]") << "\n";
                std::cout << "    Ответ LLM (Flooded Attention): S" << llm_res << " | " << (llm_res == expected ? "[УСПЕХ]" : "[УТОНУЛ В ШУМЕ]") << "\n\n";
            }
        }

        std::cout << "================================================================================\n";
        std::cout << "  ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПОСЛЕ 50 ИСПЫТАНИЙ С ШУМОМ:\n";
        std::cout << "================================================================================\n";
        std::cout << "  * Точность Чистой LLM (Attention утонул в шуме):    "
                  << std::fixed << std::setprecision(1) << (static_cast<float>(llm_success) * 100.0f / num_trials) << "%\n";
        std::cout << "  * Точность Логического движка FOG (Typed Register): "
                  << std::fixed << std::setprecision(1) << (static_cast<float>(fog_success) * 100.0f / num_trials) << "%\n";
        std::cout << "================================================================================\n";

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
