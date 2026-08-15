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
        } else {
            std::cerr << "Error: checkpoint not found: " << checkpoint << "\n";
            return 1;
        }

        install_state_codebook(model, n_states);
        motifcl::autograd::NoGradGuard no_grad;

        auto norm_weight = motifcl::Tensor::ones(backend, {cfg.d_model}, motifcl::DType::F32);
        norm_weight.set_requires_grad(false);
        auto codebook = motifcl::rmsnorm(model.lexical.token_embedding.weight.data, norm_weight);
        codebook.set_requires_grad(false);

        struct Scenario {
            std::string prompt_text;
            std::string question_text;
            std::string entity_name;
            std::vector<std::string> state_names;
            int initial_state;
            std::vector<int> step_operands; // operand added at each step modulo 16
        };

        std::vector<Scenario> scenarios = {
            {
                "СЦЕНАРИЙ 1: Отслеживание золотого ключа по комнатам замка",
                "Вопрос: В какой комнате окажется Золотой Ключ после всех перемещений?",
                "Золотой Ключ",
                {"Тронный зал (S0)", "Библиотека (S1)", "Оружейная (S2)", "Башня Мага (S3)",
                 "Лаборатория (S4)", "Подземелье (S5)", "Сокровищница (S6)", "Сад (S7)",
                 "Арсенал (S8)", "Кузница (S9)", "Винный погреб (S10)", "Галерея (S11)",
                 "Обсерватория (S12)", "Тайник (S13)", "Врата замка (S14)", "Тайный алтарь (S15)"},
                3, // Начало: Башня Мага (S3)
                {4, 5} // Шаг 1: +4 -> Сокровищница (S7), Шаг 2: +5 -> Обсерватория (S12)
            },
            {
                "СЦЕНАРИЙ 2: Изменение статуса защитной системы базы",
                "Вопрос: Какой финальный статус защитной системы после 4 команд оператора?",
                "Защитная система",
                {"ОТКЛЮЧЕНА (S0)", "ДЕЖУРНЫЙ РЕЖИМ (S1)", "СКАНИРОВАНИЕ (S2)", "ТРЕВОГА УРОВЕНЬ 1 (S3)",
                 "ТРЕВОГА УРОВЕНЬ 2 (S4)", "БЛОКИРОВКА ШЛЮЗОВ (S5)", "ПОЛНАЯ БРОНЯ (S6)", "ЗАЩИТНЫЙ КУПОЛ (S7)",
                 "РЕЖИМ ПЕРЕГРУЗКИ (S8)", "ОХЛАЖДЕНИЕ ЯДРА (S9)", "ПЕРЕЗАГРУЗКА (S10)", "БОЕВАЯ ГОТОВНОСТЬ (S11)",
                 "РЕЖИМ МАСКИРОВКИ (S12)", "АВАРИЙНЫЙ ЩИТ (S13)", "АВТОНОМНЫЙ ПАТРУЛЬ (S14)", "МАКСИМАЛЬНЫЙ РЕЖИМ (S15)"},
                1, // Начало: ДЕЖУРНЫЙ РЕЖИМ (S1)
                {2, 4, 3, 1} // 1 -> (+2)=3 -> (+4)=7 -> (+3)=10 -> (+1)=11 (БОЕВАЯ ГОТОВНОСТЬ)
            },
            {
                "СЦЕНАРИЙ 3: Длинный 8-шаговый маршрут автономного дрона-исследователя",
                "Вопрос: В каком секторе финиширует дрон после 8 навигационных переходов?",
                "Дрон-исследователь",
                {"Сектор Альфа (S0)", "Сектор Бета (S1)", "Сектор Гамма (S2)", "Сектор Дельта (S3)",
                 "Сектор Эпсилон (S4)", "Сектор Дзета (S5)", "Сектор Эта (S6)", "Сектор Тета (S7)",
                 "Сектор Йота (S8)", "Сектор Каппа (S9)", "Сектор Лямбда (S10)", "Сектор Мю (S11)",
                 "Сектор Ню (S12)", "Сектор Кси (S13)", "Сектор Омикрон (S14)", "Сектор Омега (S15)"},
                0, // Начало: Сектор Альфа (S0)
                {1, 2, 3, 1, 2, 1, 3, 2} // 0 -> 1 -> 3 -> 6 -> 7 -> 9 -> 10 -> 13 -> 15 (Сектор Омега)
            }
        };

        for (const auto& sc : scenarios) {
            std::cout << "================================================================================\n";
            std::cout << "  " << sc.prompt_text << "\n";
            std::cout << "================================================================================\n";
            std::cout << "  * Объект наблюдения: " << sc.entity_name << "\n";
            std::cout << "  * Начальная локация:  " << sc.state_names[static_cast<std::size_t>(sc.initial_state)] << "\n";
            std::cout << "  * " << sc.question_text << "\n\n";

            int cur = sc.initial_state;
            const int depth = static_cast<int>(sc.step_operands.size());
            std::vector<int> step_targets;
            for (int t = 0; t < depth; ++t) {
                cur = (cur + sc.step_operands[static_cast<std::size_t>(t)]) % n_states;
                step_targets.push_back(cur);
            }

            // Init machine with initial query token
            std::int32_t q_tok = 4 + sc.initial_state;
            auto q_tensor = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &q_tok);
            auto reg_state = model.initial_state(q_tensor);

            std::cout << "  [ РАБОТА ЛОГИЧЕСКОГО ПРОЦЕССОРА FOG В VRAM GPU RX 580 ]\n";
            std::cout << "  ------------------------------------------------------------------------------\n";

            int running_state = sc.initial_state;
            bool all_ok = true;

            for (int t = 0; t < depth; ++t) {
                const int delta = sc.step_operands[static_cast<std::size_t>(t)];
                std::vector<std::int32_t> k_ids(static_cast<std::size_t>(n_states));
                std::vector<std::int32_t> v_ids(static_cast<std::size_t>(n_states));
                for (int s = 0; s < n_states; ++s) {
                    k_ids[static_cast<std::size_t>(s)] = 4 + s;
                    v_ids[static_cast<std::size_t>(s)] = 4 + delta;
                }
                auto k_tensor = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, k_ids.data());
                auto v_tensor = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, v_ids.data());

                const auto t0 = std::chrono::steady_clock::now();
                auto out = model.structured_step(reg_state, k_tensor, v_tensor, 1, n_states);
                reg_state = out.state;
                backend.finish();
                const auto t1 = std::chrono::steady_clock::now();
                const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

                const int decoded = decode_state(reg_state.value, codebook, norm_weight, n_states);
                const int expected = step_targets[static_cast<std::size_t>(t)];

                auto op_l = out.operator_logits.to_vector<float>();
                int best_op = 0;
                for (int o = 1; o < 7; ++o) {
                    if (op_l[static_cast<std::size_t>(o)] > op_l[static_cast<std::size_t>(best_op)]) best_op = o;
                }

                const bool match = (decoded == expected);
                if (!match) all_ok = false;

                std::cout << "  Такт " << (t + 1) << "/" << depth
                          << " (" << std::fixed << std::setprecision(1) << us << " мкс): "
                          << sc.state_names[static_cast<std::size_t>(running_state)]
                          << " -> " << sc.state_names[static_cast<std::size_t>(decoded)]
                          << " | Оператор: " << std::setw(13) << std::left << kOpNames[best_op]
                          << " | " << (match ? "[ТОЧНО]" : "[ОШИБКА]") << "\n";

                running_state = decoded;
            }

            std::cout << "  ------------------------------------------------------------------------------\n";
            std::cout << "  ОТВЕТ МОДЕЛИ:  " << sc.entity_name << " находится в локации: «"
                      << sc.state_names[static_cast<std::size_t>(running_state)] << "»\n";
            std::cout << "  ЭТАЛОН:        " << sc.state_names[static_cast<std::size_t>(step_targets.back())] << "\n";
            std::cout << "  ВЕРДИКТ:       " << (all_ok ? "ИДЕАЛЬНО (100% точность, 0 галлюцинаций)" : "ОШИБКА") << "\n\n";
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
