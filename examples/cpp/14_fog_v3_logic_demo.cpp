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
const char* const kOpNames[7] = {
    "IDENTITY",
    "ADD_INVERSE",
    "BLOCK_PRODUCT",
    "SWAP",
    "CLEAR",
    "ADD",
    "PROJECT"
};

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
        if (std::filesystem::exists(checkpoint)) {
            motifcl::load_parameters(params, backend, checkpoint);
            std::cout << "[OK] Loaded checkpoint: " << checkpoint << "\n\n";
        } else {
            std::cerr << "Error: checkpoint not found: " << checkpoint << "\n";
            return 1;
        }

        install_state_codebook(model, n_states);
        motifcl::autograd::NoGradGuard no_grad;

        // Precompute normalized codebook
        auto norm_weight = motifcl::Tensor::ones(backend, {cfg.d_model}, motifcl::DType::F32);
        norm_weight.set_requires_grad(false);
        auto codebook = motifcl::rmsnorm(model.lexical.token_embedding.weight.data, norm_weight);
        codebook.set_requires_grad(false);

        struct TestPuzzle {
            std::string title;
            int initial_state;
            std::vector<int> table; // table of operands for each state 0..15
            int depth;
        };

        std::mt19937 rng(42u);
        std::uniform_int_distribution<int> sdist(0, n_states - 1);

        auto make_puzzle = [&](const std::string& title, int init_s, int d) {
            TestPuzzle pz;
            pz.title = title;
            pz.initial_state = init_s;
            pz.depth = d;
            pz.table.resize(static_cast<std::size_t>(n_states));
            for (int s = 0; s < n_states; ++s) pz.table[static_cast<std::size_t>(s)] = sdist(rng);
            return pz;
        };

        std::vector<TestPuzzle> puzzles = {
            make_puzzle("Задача 1 (2 шага): Двухшаговая цепочка ассоциативных переходов", 3, 2),
            make_puzzle("Задача 2 (4 шага): Четырёхшаговый поиск в графе состояний", 7, 4),
            make_puzzle("Задача 3 (8 шагов): Глубокая 8-шаговая рекурсия (В 2 раза глубже обучения!)", 1, 8)
        };

        for (const auto& pz : puzzles) {
            std::cout << "======================================================================\n";
            std::cout << "  " << pz.title << "\n";
            std::cout << "======================================================================\n";
            std::cout << "  * Начальное состояние: State S" << pz.initial_state << "\n";

            int cur = pz.initial_state;
            const int depth = pz.depth;

            // Build transition table (keys = 0..n_states-1, values = pz.table[s])
            std::vector<int> step_targets;
            for (int t = 0; t < depth; ++t) {
                const int operand = pz.table[static_cast<std::size_t>(cur)];
                cur = (cur + operand) % n_states;
                step_targets.push_back(cur);
            }
            std::cout << "  * Ожидаемый итоговый ответ (Ground Truth): State S" << cur << "\n";
            std::cout << "  --------------------------------------------------------------------\n";

            // Init machine with initial query token
            std::int32_t q_tok = 4 + pz.initial_state;
            auto q_tensor = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &q_tok);
            auto reg_state = model.initial_state(q_tensor);

            int running_state = pz.initial_state;
            bool all_steps_correct = true;

            std::vector<std::int32_t> k_ids(static_cast<std::size_t>(n_states));
            std::vector<std::int32_t> v_ids(static_cast<std::size_t>(n_states));
            for (int s = 0; s < n_states; ++s) {
                k_ids[static_cast<std::size_t>(s)] = 4 + s;
                v_ids[static_cast<std::size_t>(s)] = 4 + pz.table[static_cast<std::size_t>(s)];
            }
            auto k_tensor = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, k_ids.data());
            auto v_tensor = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, v_ids.data());

            for (int t = 0; t < depth; ++t) {
                auto out = model.structured_step(reg_state, k_tensor, v_tensor, 1, n_states);
                reg_state = out.state;

                // Decode current register state
                const int decoded = decode_state(reg_state.value, codebook, norm_weight, n_states);
                const int expected = step_targets[static_cast<std::size_t>(t)];

                // Get operator logits
                auto op_l = out.operator_logits.to_vector<float>();
                int best_op = 0;
                for (int o = 1; o < 7; ++o) {
                    if (op_l[static_cast<std::size_t>(o)] > op_l[static_cast<std::size_t>(best_op)]) best_op = o;
                }

                auto halt_v = out.halt_probability.to_vector<float>();
                const float halt = halt_v.empty() ? 0.0f : halt_v[0];
                const bool match = (decoded == expected);
                if (!match) all_steps_correct = false;

                std::cout << "  Такт " << (t + 1) << "/" << depth
                          << " | Переход: S" << running_state << " -> S" << expected
                          << " | Регистр R_val: S" << decoded
                          << " | Оператор: " << std::setw(13) << std::left << kOpNames[best_op]
                          << " | HALT: " << std::fixed << std::setprecision(2) << halt
                          << " | " << (match ? "[OK]" : "[FAIL]") << "\n";
                running_state = decoded;
            }

            std::cout << "  --------------------------------------------------------------------\n";
            std::cout << "  ИТОГ ВЫЧИСЛЕНИЙ В ПАМЯТИ: " << (all_steps_correct ? "УСПЕХ (100% точность)" : "ОШИБКА")
                      << " | Ответ: State S" << running_state << "\n\n";
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Logic demo error: " << exc.what() << "\n";
        return 1;
    }
}
