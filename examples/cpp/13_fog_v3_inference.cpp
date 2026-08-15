#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {
int sample_top_k(const float* logits, int vocab, float temperature, int top_k, std::mt19937& rng) {
    if (temperature <= 1e-4f) {
        int best = 0;
        for (int i = 1; i < vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    struct Candidate {
        int id;
        float val;
    };
    std::vector<Candidate> cands(static_cast<std::size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        cands[static_cast<std::size_t>(i)] = {i, logits[i] / temperature};
    }

    top_k = std::min(top_k, vocab);
    std::partial_sort(cands.begin(), cands.begin() + top_k, cands.end(), [](const Candidate& a, const Candidate& b) {
        return a.val > b.val;
    });

    float max_val = cands[0].val;
    float sum_exp = 0.0f;
    std::vector<float> probs(static_cast<std::size_t>(top_k));
    for (int i = 0; i < top_k; ++i) {
        probs[static_cast<std::size_t>(i)] = std::exp(cands[static_cast<std::size_t>(i)].val - max_val);
        sum_exp += probs[static_cast<std::size_t>(i)];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum_exp);
    const float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        acc += probs[static_cast<std::size_t>(i)];
        if (r <= acc) return cands[static_cast<std::size_t>(i)].id;
    }
    return cands[0].id;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string checkpoint = "checkpoints/fog_v3_rx580_reasoning_sft.mclp";
        int max_tokens = 64;
        float temperature = 0.7f;
        int top_k = 40;
        std::vector<std::int32_t> prompt_ids;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--checkpoint" && i + 1 < argc) checkpoint = argv[++i];
            else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
            else if (arg == "--temperature" && i + 1 < argc) temperature = std::strtof(argv[++i], nullptr);
            else if (arg == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
            else if (arg == "--prompt-ids" && i + 1 < argc) {
                std::string s = argv[++i];
                std::stringstream ss(s);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (!item.empty()) prompt_ids.push_back(std::atoi(item.c_str()));
                }
            }
        }

        if (prompt_ids.empty()) {
            prompt_ids = {1, 284, 1500}; // fallback prompt
        }

        auto backend = motifcl::Backend::create_vulkan();
        motifcl::nn::FogV3Config cfg;
        cfg.vocab_size = 8192;
        cfg.max_seq_len = 512;
        cfg.d_model = 320;
        cfg.n_heads = 5;
        cfg.n_layers = 4;
        cfg.d_ff = 1344;
        cfg.dropout = 0.0f;
        motifcl::nn::FogV3Model model(backend, cfg);

        if (std::filesystem::exists(checkpoint)) {
            try {
                auto params = model.parameters();
                motifcl::load_parameters(params, backend, checkpoint);
            } catch (const std::exception&) {
                auto lex_params = model.lexical.parameters();
                motifcl::load_parameters(lex_params, backend, checkpoint);
            }
        } else {
            throw std::runtime_error("Checkpoint not found: " + checkpoint);
        }

        std::mt19937 rng(static_cast<std::uint32_t>(std::chrono::system_clock::now().time_since_epoch().count()));
        motifcl::autograd::NoGradGuard no_grad;

        std::vector<std::int32_t> generated_ids = prompt_ids;
        auto& runtime = backend.vulkan_runtime();

        for (int step = 0; step < max_tokens; ++step) {
            const int seq_len = static_cast<int>(generated_ids.size());
            if (seq_len >= cfg.max_seq_len) break;

            auto tokens = motifcl::Tensor::from_cpu(backend, {1, seq_len}, motifcl::DType::I32, generated_ids.data());
            const bool batched = runtime.batch_begin();
            auto logits3 = model.forward(tokens);
            if (batched) {
                const auto submit = runtime.batch_end();
                if (!submit.success) throw std::runtime_error("Inference submit failed: " + submit.error);
            }
            backend.finish();

            auto lhost = logits3.to_vector<float>();
            const float* last_logits = lhost.data() + (seq_len - 1) * cfg.vocab_size;
            const int next_token = sample_top_k(last_logits, cfg.vocab_size, temperature, top_k, rng);
            generated_ids.push_back(next_token);

            if (next_token == 2) break; // <|endoftext|> / EOS
        }

        // Print generated token IDs as comma-separated list
        std::cout << "OUTPUT_IDS:";
        for (std::size_t i = 0; i < generated_ids.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << generated_ids[i];
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Inference error: " << exc.what() << "\n";
        return 1;
    }
}
