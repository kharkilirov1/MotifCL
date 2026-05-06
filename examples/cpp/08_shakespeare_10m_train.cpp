#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

namespace {

std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw std::runtime_error("failed to open dataset: " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::size_t count_parameters(const std::vector<motifcl::nn::Parameter*>& params) {
    std::size_t total = 0;
    for (auto* p : params) {
        if (p) total += static_cast<std::size_t>(p->data.numel());
    }
    return total;
}

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::string(value) != "0";
}

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    return std::max(1, std::atoi(value));
}

void print_profile_summary(const motifcl::Backend& backend, int step, int top_n) {
    const auto summary = backend.profiler.summary();
    double total_ms = 0.0;
    for (const auto& item : summary) total_ms += item.total_ms;
    std::cout << "profile step " << step << " top " << std::min<int>(top_n, static_cast<int>(summary.size()))
              << " kernel/transfer groups, profiled_ms=" << total_ms << "\n";
    std::cout << std::left << std::setw(34) << "name"
              << std::right << std::setw(10) << "count"
              << std::setw(14) << "total_ms"
              << std::setw(12) << "avg_ms"
              << std::setw(12) << "max_ms" << "\n";
    for (int i = 0; i < static_cast<int>(summary.size()) && i < top_n; ++i) {
        const auto& item = summary[static_cast<std::size_t>(i)];
        std::cout << std::left << std::setw(34) << item.name
                  << std::right << std::setw(10) << item.count
                  << std::setw(14) << item.total_ms
                  << std::setw(12) << item.avg_ms
                  << std::setw(12) << item.max_ms << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string dataset_path = argc > 1 ? argv[1] : "build/datasets/tiny_shakespeare.txt";
        const int steps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 100;
        const int batch_size = argc > 3 ? std::max(1, std::atoi(argv[3])) : 1;

        constexpr int vocab_size = 256; // byte-level real-text model
        constexpr int seq_len = 128;
        constexpr int n_embd = 512;
        constexpr int n_head = 8;
        constexpr int n_layer = 3;
        constexpr int mlp_hidden = 2304;

        auto bytes = read_bytes(dataset_path);
        if (bytes.size() <= static_cast<std::size_t>(seq_len + 1)) {
            throw std::runtime_error("dataset is too small for configured sequence length");
        }

        motifcl::manual_seed(20260505);
        std::mt19937 rng(20260505);
        std::uniform_int_distribution<std::size_t> pos_dist(0, bytes.size() - static_cast<std::size_t>(seq_len) - 2);

        auto backend = motifcl::Backend::create_opencl();
        const auto info = backend.device_info();
        const bool profile = env_enabled("MOTIFCL_PROFILE");
        const int profile_top = env_int("MOTIFCL_PROFILE_TOP", 25);
        backend.profiler.set_enabled(profile);

        motifcl::nn::GPTModel model(backend, vocab_size, seq_len, n_embd, n_head, n_layer, mlp_hidden);
        auto params = model.parameters();
        const std::size_t param_count = count_parameters(params);
        motifcl::optim::Adam opt(params, 3e-4f);

        std::cout << "dataset=" << dataset_path << " bytes=" << bytes.size() << "\n";
        std::cout << "device=" << info.device_name << " driver=" << info.driver_version << "\n";
        if (profile) std::cout << "operator profiler enabled; set MOTIFCL_PROFILE_TOP=N to change output size\n";
        std::cout << "model: vocab=" << vocab_size
                  << " seq_len=" << seq_len
                  << " batch=" << batch_size
                  << " n_embd=" << n_embd
                  << " n_head=" << n_head
                  << " n_layer=" << n_layer
                  << " mlp_hidden=" << mlp_hidden
                  << " params=" << param_count
                  << " (~" << (static_cast<double>(param_count) / 1.0e6) << "M)\n";

        std::vector<std::int32_t> x_host(static_cast<std::size_t>(batch_size * seq_len));
        std::vector<std::int32_t> y_host(static_cast<std::size_t>(batch_size * seq_len));

        double total_ms = 0.0;
        double measured_ms = 0.0;
        int measured_steps = 0;
        float first_loss = 0.0f;
        float last_loss = 0.0f;

        for (int step = 1; step <= steps; ++step) {
            if (profile) backend.profiler.clear();
            for (int b = 0; b < batch_size; ++b) {
                const std::size_t start = pos_dist(rng);
                for (int t = 0; t < seq_len; ++t) {
                    x_host[static_cast<std::size_t>(b * seq_len + t)] = static_cast<std::int32_t>(bytes[start + static_cast<std::size_t>(t)]);
                    y_host[static_cast<std::size_t>(b * seq_len + t)] = static_cast<std::int32_t>(bytes[start + static_cast<std::size_t>(t + 1)]);
                }
            }

            const auto t0 = std::chrono::steady_clock::now();
            auto x = motifcl::Tensor::from_cpu(backend, {batch_size, seq_len}, motifcl::DType::I32, x_host.data());
            auto y = motifcl::Tensor::from_cpu(backend, {batch_size * seq_len}, motifcl::DType::I32, y_host.data());
            auto logits3 = model.forward(x);
            auto logits = logits3.view({batch_size * seq_len, vocab_size});
            auto loss = motifcl::softmax_cross_entropy(logits, y);
            loss.backward();
            opt.step();
            opt.zero_grad();
            backend.finish();
            const auto t1 = std::chrono::steady_clock::now();

            const double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += step_ms;
            if (step > 1) {
                measured_ms += step_ms;
                ++measured_steps;
            }
            const float loss_value = loss.item();
            if (step == 1) first_loss = loss_value;
            last_loss = loss_value;
            if (profile && (step == 1 || step == steps || step % 10 == 0)) {
                print_profile_summary(backend, step, profile_top);
            }

            if (step == 1 || step % 10 == 0 || step == steps) {
                const double tokens_per_sec = (1000.0 * static_cast<double>(batch_size * seq_len)) / step_ms;
                std::cout << "step " << step
                          << "/" << steps
                          << " loss=" << loss_value
                          << " step_ms=" << step_ms
                          << " tok/s=" << tokens_per_sec
                          << "\n";
            }
        }

        const double avg_ms = total_ms / static_cast<double>(steps);
        const double avg_measured_ms = measured_steps > 0 ? measured_ms / static_cast<double>(measured_steps) : avg_ms;
        const double tokens_per_sec = (1000.0 * static_cast<double>(batch_size * seq_len)) / avg_measured_ms;
        std::cout << "summary: steps=" << steps
                  << " first_loss=" << first_loss
                  << " last_loss=" << last_loss
                  << " avg_step_ms=" << avg_ms
                  << " avg_step_ms_excluding_first=" << avg_measured_ms
                  << " tokens_per_sec_excluding_first=" << tokens_per_sec
                  << "\n";
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "08_shakespeare_10m_train");
    }
}
