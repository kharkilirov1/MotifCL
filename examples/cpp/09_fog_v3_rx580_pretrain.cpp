#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

namespace {

std::vector<std::int32_t> read_i32(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw std::runtime_error("failed to open token file: " + path);
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (bytes % sizeof(std::int32_t) != 0) throw std::runtime_error("token file size is not a multiple of int32");
    std::vector<std::int32_t> out(bytes / sizeof(std::int32_t));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
}

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

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string token_path = argc > 1 ? argv[1] : "data/fog_train.i32";
        const int steps = arg_int(argv, argc, 2, 1000);
        const int batch = arg_int(argv, argc, 3, 8);
        const int seq = arg_int(argv, argc, 4, 128);
        const float lr = arg_float(argv, argc, 5, 3e-4f);
        const std::string checkpoint = argc > 6 ? argv[6] : "checkpoints/fog_v3_rx580_lexical.mclp";
        const int save_every = arg_int(argv, argc, 7, 250);
        const std::string resume_path = argc > 8 ? argv[8] : checkpoint;

        constexpr int vocab = 8192;
        auto tokens = read_i32(token_path);
        if (tokens.size() <= static_cast<std::size_t>(seq + 1)) {
            throw std::runtime_error("token file is too small for requested sequence length");
        }
        for (auto id : tokens) {
            if (id < 0 || id >= vocab) throw std::runtime_error("token id outside vocab=8192");
        }

        auto backend = motifcl::Backend::create_vulkan();
        const auto info = backend.device_info();
        motifcl::manual_seed(20260815u);
        std::mt19937 rng(20260815u);
        std::uniform_int_distribution<std::size_t> start_dist(
            0, tokens.size() - static_cast<std::size_t>(seq) - 2);

        motifcl::nn::FogV3Config cfg;
        cfg.vocab_size = vocab;
        cfg.max_seq_len = std::max(512, seq);
        cfg.d_model = 320;
        cfg.n_heads = 5;
        cfg.n_layers = 4;
        cfg.d_ff = 1344;
        cfg.dropout = 0.0f;
        motifcl::nn::FogV3Model model(backend, cfg);
        auto params = model.parameters();
        auto train_params = model.lexical_parameters();

        if (std::filesystem::exists(resume_path)) {
            motifcl::load_parameters(params, backend, resume_path);
            std::cout << "resume weights=" << resume_path << " (Adam moments restart)\n";
        }
        motifcl::optim::Adam opt(train_params, lr, 0.9f, 0.95f, 1e-8f, 0.01f);

        std::filesystem::create_directories(std::filesystem::path(checkpoint).parent_path());
        std::cout << "FOG v3 RX580 Vulkan lexical pretrain\n"
                  << "device=" << info.device_name << " driver=" << info.driver_version << "\n"
                  << "tokens=" << tokens.size() << " vocab=" << vocab
                  << " params=" << parameter_count(params)
                  << " batch=" << batch << " seq=" << seq << " lr=" << lr << "\n";

        std::vector<std::int32_t> xh(static_cast<std::size_t>(batch * seq));
        std::vector<std::int32_t> yh(static_cast<std::size_t>(batch * seq));
        double elapsed = 0.0;
        float first_loss = 0.0f;
        float last_loss = 0.0f;

        const int warmup_steps = std::min(steps / 20, 500);
        for (int step = 1; step <= steps; ++step) {
            float cur_lr = lr;
            if (step <= warmup_steps) {
                cur_lr = lr * static_cast<float>(step) / static_cast<float>(std::max(1, warmup_steps));
            } else {
                const float progress = static_cast<float>(step - warmup_steps) / static_cast<float>(std::max(1, steps - warmup_steps));
                cur_lr = lr * (0.1f + 0.9f * 0.5f * (1.0f + std::cos(3.14159265358979323846f * progress)));
            }
            opt.set_lr(cur_lr);

            for (int b = 0; b < batch; ++b) {
                const auto start = start_dist(rng);
                for (int t = 0; t < seq; ++t) {
                    xh[static_cast<std::size_t>(b * seq + t)] = tokens[start + static_cast<std::size_t>(t)];
                    yh[static_cast<std::size_t>(b * seq + t)] = tokens[start + static_cast<std::size_t>(t + 1)];
                }
            }

            const auto t0 = std::chrono::steady_clock::now();
            auto x = motifcl::Tensor::from_cpu(backend, {batch, seq}, motifcl::DType::I32, xh.data());
            auto y = motifcl::Tensor::from_cpu(backend, {batch * seq}, motifcl::DType::I32, yh.data());

            auto& runtime = backend.vulkan_runtime();
            const bool batched = runtime.batch_begin();
            auto logits3 = model.forward(x);
            auto logits = logits3.view({batch * seq, vocab});
            auto loss = motifcl::softmax_cross_entropy(logits, y);
            loss.backward();
            opt.step();
            opt.zero_grad();
            if (batched) {
                const auto submit = runtime.batch_end();
                if (!submit.success) throw std::runtime_error("Vulkan train batch failed: " + submit.error);
            }
            backend.finish();
            const float lv = loss.item();
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            elapsed += ms;
            if (step == 1) first_loss = lv;
            last_loss = lv;

            if (step == 1 || step % 50 == 0 || step == steps) {
                const double tps = 1000.0 * static_cast<double>(batch * seq) / ms;
                std::cout << "step=" << step << "/" << steps
                          << " loss=" << std::setprecision(6) << lv
                          << " lr=" << std::scientific << std::setprecision(2) << cur_lr << std::defaultfloat
                          << " ms=" << std::setprecision(5) << ms
                          << " tok/s=" << std::setprecision(6) << tps << "\n";
            }
            if (step % save_every == 0 || step == steps) {
                motifcl::save_parameters(params, checkpoint);
                std::cout << "saved=" << checkpoint << " step=" << step << "\n";
            }
        }
        std::cout << "done first_loss=" << first_loss << " last_loss=" << last_loss
                  << " avg_ms=" << elapsed / static_cast<double>(steps) << "\n";
        return 0;
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "09_fog_v3_rx580_pretrain");
    }
}
