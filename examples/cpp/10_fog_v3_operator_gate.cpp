#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

namespace {

constexpr int kMod = 31;
constexpr float kPi = 3.14159265358979323846f;

int arg_int(char** argv, int argc, int idx, int fallback) {
    if (idx >= argc) return fallback;
    return std::max(1, std::atoi(argv[idx]));
}

float arg_float(char** argv, int argc, int idx, float fallback) {
    if (idx >= argc) return fallback;
    return std::max(1e-8f, std::strtof(argv[idx], nullptr));
}

std::vector<float> code(int x, int d_model) {
    std::vector<float> out(static_cast<std::size_t>(d_model));
    const float amp = std::sqrt(2.0f);
    const int pairs = d_model / 2;
    for (int j = 0; j < pairs; ++j) {
        const int harmonic = (j % (kMod - 1)) + 1;
        const float angle = 2.0f * kPi * static_cast<float>(harmonic * x) / static_cast<float>(kMod);
        out[static_cast<std::size_t>(2 * j)] = amp * std::cos(angle);
        out[static_cast<std::size_t>(2 * j + 1)] = amp * std::sin(angle);
    }
    return out;
}

void fill_batch(std::vector<float>& value,
                std::vector<float>& addressed,
                std::vector<float>& target,
                int batch,
                int d_model,
                std::mt19937& rng) {
    std::uniform_int_distribution<int> state_dist(0, kMod - 1);
    for (int r = 0; r < batch; ++r) {
        const int a = state_dist(rng);
        const int b = state_dist(rng);
        const auto va = code(a, d_model);
        const auto vb = code(b, d_model);
        const auto vt = code((a + b) % kMod, d_model);
        const std::size_t off = static_cast<std::size_t>(r * d_model);
        std::copy(va.begin(), va.end(), value.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(vb.begin(), vb.end(), addressed.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(vt.begin(), vt.end(), target.begin() + static_cast<std::ptrdiff_t>(off));
    }
}

float cosine_row(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        aa += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        bb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (aa <= 0.0 || bb <= 0.0) return 0.0f;
    return static_cast<float>(dot / std::sqrt(aa * bb));
}

int argmax7(const float* p) {
    int best = 0;
    for (int k = 1; k < 7; ++k) if (p[k] > p[best]) best = k;
    return best;
}

} // namespace

int main(int argc, char** argv) {
    using namespace motifcl;
    try {
        const int steps = arg_int(argv, argc, 1, 200);
        const int batch = arg_int(argv, argc, 2, 32);
        const float lr = arg_float(argv, argc, 3, 3e-3f);
        const int d_model = arg_int(argv, argc, 4, 320);
        const int depth = arg_int(argv, argc, 5, 8);
        if ((d_model % 2) != 0) throw std::runtime_error("d_model must be even");

        const auto probe = probe_vulkan_runtime();
        if (!probe.available()) {
            std::cerr << "No Vulkan compute device: " << probe.error << "\n";
            return 77;
        }
        auto backend = Backend::create_vulkan();
        const auto info = backend.device_info();
        manual_seed(20260815u);
        std::mt19937 rng(20260815u);

        nn::FogOperatorBankV3 bank(backend, d_model, 48);
        auto params = bank.parameters();
        optim::Adam opt(params, lr, 0.9f, 0.95f, 1e-8f, 0.0f);

        std::vector<float> value_h(static_cast<std::size_t>(batch * d_model));
        std::vector<float> read_h(value_h.size());
        std::vector<float> target_h(value_h.size());
        std::vector<float> control_h(value_h.size(), 0.0f);

        std::cout << "FOG v3 Vulkan operator-gate\n"
                  << "device=" << info.device_name << " driver=" << info.driver_version << "\n"
                  << "steps=" << steps << " batch=" << batch << " d_model=" << d_model
                  << " lr=" << lr << " eval_depth=" << depth << "\n";

        float first_loss = 0.0f;
        float last_loss = 0.0f;
        for (int step = 1; step <= steps; ++step) {
            fill_batch(value_h, read_h, target_h, batch, d_model, rng);
            auto value = Tensor::from_cpu(backend, {batch, d_model}, DType::F32, value_h.data());
            auto read = Tensor::from_cpu(backend, {batch, d_model}, DType::F32, read_h.data());
            auto control = Tensor::from_cpu(backend, {batch, d_model}, DType::F32, control_h.data());
            auto target = Tensor::from_cpu(backend, {batch, d_model}, DType::F32, target_h.data());

            auto out = bank.forward3(value, read, control);
            auto loss = mse_loss(out.value, target);
            loss.backward();
            opt.step();
            opt.zero_grad();
            backend.finish();
            const float lv = loss.item();
            if (step == 1) first_loss = lv;
            last_loss = lv;

            if (step == 1 || step % 20 == 0 || step == steps) {
                const auto logits_h = out.logits.to_vector<float>();
                int block = 0;
                for (int r = 0; r < batch; ++r) {
                    if (argmax7(logits_h.data() + static_cast<std::size_t>(r * 7)) == 2) ++block;
                }
                std::cout << "step=" << step << "/" << steps
                          << " loss=" << std::setprecision(7) << lv
                          << " block_route=" << block << "/" << batch << "\n";
            }
        }

        // Long-horizon witness: repeated generated state is fed back without
        // snap/decode. A single operand is used so the expected state is exact.
        const int a0 = 7;
        const int operand = 11;
        auto state_h = code(a0, d_model);
        const auto operand_h = code(operand, d_model);
        std::vector<float> zero_h(static_cast<std::size_t>(d_model), 0.0f);
        int block_steps = 0;
        float min_cos = 1.0f;
        for (int t = 1; t <= depth; ++t) {
            auto state = Tensor::from_cpu(backend, {1, d_model}, DType::F32, state_h.data());
            auto read = Tensor::from_cpu(backend, {1, d_model}, DType::F32, operand_h.data());
            auto control = Tensor::from_cpu(backend, {1, d_model}, DType::F32, zero_h.data());
            auto out = bank.forward3(state, read, control);
            const auto logits_h = out.logits.to_vector<float>();
            if (argmax7(logits_h.data()) == 2) ++block_steps;
            state_h = out.value.to_vector<float>();
            const auto expected = code((a0 + t * operand) % kMod, d_model);
            min_cos = std::min(min_cos, cosine_row(state_h, expected));
        }

        std::cout << "operator_gate_summary first_loss=" << first_loss
                  << " last_loss=" << last_loss
                  << " recurrent_block_route=" << block_steps << "/" << depth
                  << " min_cos=" << std::setprecision(8) << min_cos << "\n";

        // Conservative hardware witness thresholds: this is a gate, not a
        // benchmark. Failure means do not start a long FOG run yet.
        if (block_steps != depth || min_cos < 0.999f) {
            std::cerr << "FOG operator gate FAILED: hard grammar did not learn stable BLOCK_PRODUCT\n";
            return 2;
        }
        std::cout << "FOG operator gate PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "10_fog_v3_operator_gate");
    }
}
