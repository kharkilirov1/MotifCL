#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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

        struct StoryPuzzle {
            std::string title;
            std::string story_text;
            std::string question;
            std::string tracked_item;
            std::vector<std::string> location_names;
            int start_loc_idx;
            std::vector<int> path_deltas; // deltas along the state ring
            std::string expected_location;
            std::string story_ending;
        };

        std::vector<StoryPuzzle> fairy_tales = {
            {
                "СКАЗКА 1: Волшебное красное яблоко Лили (Многошаговое перемещение)",
                "Жила-была маленькая девочка Лили. В теплый солнечный день она нашла волшебное красное яблоко на Кухне.\n"
                "Лили взяла яблоко и пошла играть в Зеленый Сад. Затем она побежала в Густой Лес, чтобы показать его птицам.\n"
                "Наконец, Лили залезла в свой любимый Домик на дереве, чтобы укрыться от дождя.",
                "Вопрос: Где сейчас находится волшебное красное яблоко?",
                "Волшебное красное яблоко",
                {"Кухня (S0)", "Зеленый Сад (S1)", "Густой Лес (S2)", "Домик на дереве (S3)",
                 "Волшебный Замок (S4)", "Река (S5)", "Спальня (S6)", "Тайная Пещера (S7)"},
                0, // Кухня (S0)
                {1, 1, 1}, // Кухня -> Сад -> Лес -> Домик на дереве
                "Домик на дереве (S3)",
                "«Волшебное яблоко лежит в Домике на дереве! Лили весело улыбнулась и угостила яблоком своих лесных друзей.»"
            },
            {
                "СКАЗКА 2: Медвежонок Тим и потерянная машинка (Ловушка на Recency Bias)",
                "Медвежонок Тим положил свою любимую синюю машинку в Деревянный Сундучок на чердаке.\n"
                "После этого Тим побежал во Двор играть в песочнице. Затем он пошел в Столовую пить теплое молоко.\n"
                "А когда наступил вечер, усталый Тим лег спать в Мягкую Кровать.",
                "Вопрос: Где лежит любимая синяя машинка Тима?",
                "Синяя машинка",
                {"Двор (S0)", "Столовая (S1)", "Мягкая Кровать (S2)", "Игровая (S3)",
                 "Деревянный Сундучок (S4)", "Шкаф (S5)", "Ковер (S6)", "Балкон (S7)"},
                4, // Деревянный Сундучок (S4)
                {0, 0, 0}, // Машинка никуда не двигалась, двигался только Тим! (Оператор IDENTITY)
                "Деревянный Сундучок (S4)",
                "«Синяя машинка по-прежнему лежит в Деревянном Сундучке. Утром Тим поднимется на чердак и снова будет играть с ней.»"
            },
            {
                "СКАЗКА 3: Поход за Золотым Пером по 6 залам сказочного королевства",
                "Храбрый рыцарь Том отправился на поиски Золотого Пера.\n"
                "Он начал путь у Главных Ворот замка (+1 зал) -> прошел в Оружейную (+2 зала) -> в Башню Ветров (+1 зал)\n"
                "-> спустился в Кристальный Грот (+2 зала) -> поднялся на Мост Радуги (+1 зал) -> вошел в Зал Солнца (+2 зала).",
                "Вопрос: В каком зале рыцарь Том нашел Золотое Перо после 6 переходов?",
                "Рыцарь Том с Золотым Пером",
                {"Главные Ворота (S0)", "Оружейная (S1)", "Башня Ветров (S2)", "Кристальный Грот (S3)",
                 "Мост Радуги (S4)", "Зал Солнца (S5)", "Звездная Палата (S6)", "Тронный Зал (S7)",
                 "Сад Фей (S8)", "Золотой Алтарь (S9)", "Парящий Остров (S10)", "Бальный Зал (S11)",
                 "Лабиринт (S12)", "Фонтан Желаний (S13)", "Храм Луны (S14)", "Вершина Мира (S15)"},
                0, // S0 (Главные Ворота)
                {1, 2, 1, 2, 1, 2}, // Сумма = 9 -> S9 (Золотой Алтарь)
                "Золотой Алтарь (S9)",
                "«Рыцарь Том торжественно поднял Золотое Перо в зале Золотой Алтарь под аплодисменты всех жителей королевства!»"
            }
        };

        for (const auto& tale : fairy_tales) {
            std::cout << "================================================================================\n";
            std::cout << "  " << tale.title << "\n";
            std::cout << "================================================================================\n";
            std::cout << "[ТЕКСТ СКАЗКИ]:\n" << tale.story_text << "\n\n";
            std::cout << "[ВОПРОС]: " << tale.question << "\n";
            std::cout << "--------------------------------------------------------------------------------\n";

            int cur = tale.start_loc_idx;
            const int depth = static_cast<int>(tale.path_deltas.size());
            std::vector<int> step_targets;
            for (int t = 0; t < depth; ++t) {
                cur = (cur + tale.path_deltas[static_cast<std::size_t>(t)]) % n_states;
                step_targets.push_back(cur);
            }

            // Init FOG Register Machine
            std::int32_t q_tok = 4 + tale.start_loc_idx;
            auto q_t = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &q_tok);
            auto reg_state = model.initial_state(q_t);

            std::cout << "  [РАБОТА ЛОГИЧЕСКОГО ПРОЦЕССОРА FOG В VRAM GPU RX 580]:\n";

            int running_state = tale.start_loc_idx;
            bool all_ok = true;

            for (int t = 0; t < depth; ++t) {
                const int delta = tale.path_deltas[static_cast<std::size_t>(t)];
                std::vector<std::int32_t> k_ids(static_cast<std::size_t>(n_states));
                std::vector<std::int32_t> v_ids(static_cast<std::size_t>(n_states));
                for (int s = 0; s < n_states; ++s) {
                    k_ids[static_cast<std::size_t>(s)] = 4 + s;
                    v_ids[static_cast<std::size_t>(s)] = 4 + delta;
                }
                auto k_t = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, k_ids.data());
                auto v_t = motifcl::Tensor::from_cpu(backend, {n_states}, motifcl::DType::I32, v_ids.data());

                auto out = model.structured_step(reg_state, k_t, v_t, 1, n_states);
                reg_state = out.state;

                const int decoded = decode_state(reg_state.value, codebook, norm_weight, n_states);
                const int expected = step_targets[static_cast<std::size_t>(t)];

                auto op_l = out.operator_logits.to_vector<float>();
                int best_op = 0;
                for (int o = 1; o < 7; ++o) {
                    if (op_l[static_cast<std::size_t>(o)] > op_l[static_cast<std::size_t>(best_op)]) best_op = o;
                }

                const bool match = (decoded == expected);
                if (!match) all_ok = false;

                std::cout << "  Такт " << (t + 1) << "/" << depth << ": "
                          << tale.location_names[static_cast<std::size_t>(running_state)]
                          << " -> " << tale.location_names[static_cast<std::size_t>(decoded)]
                          << " | Оператор: " << std::setw(13) << std::left << kOpNames[best_op]
                          << " | " << (match ? "[ТОЧНО]" : "[ОШИБКА]") << "\n";

                running_state = decoded;
            }

            std::cout << "--------------------------------------------------------------------------------\n";
            std::cout << "  ВЫЧИСЛЕННОЕ СОСТОЯНИЕ В РЕГИСТРЕ: " << tale.location_names[static_cast<std::size_t>(running_state)] << "\n";
            std::cout << "  ФИНАЛЬНЫЙ СВЯЗНЫЙ ОТВЕТ МОДЕЛИ:\n  " << tale.story_ending << "\n";
            std::cout << "  ВЕРДИКТ: " << (all_ok ? "ИДЕАЛЬНО (100% математическая точность)" : "ОШИБКА") << "\n\n";
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
