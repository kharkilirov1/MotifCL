// Slice R3 witness: end-to-end memory-native training on Vulkan (no OpenCL
// context). Two independent witnesses:
//
//   (A) Single CounterStateLinear teacher-recovery on Vulkan — proves the
//       counter synapse (pillar A) trains correctly on Vulkan device-resident
//       tensors. The fused in-backward update (counter_row_stats_fused +
//       counter_apply_update_fused via Vulkan dispatch_cached) must drive MSE
//       against a ternary teacher signal toward 0 and reach >= 80% ternary
//       accuracy on the decoded weight.
//
//   (B) 4-block reversible stack with CounterStateLinear coupling on Vulkan —
//       proves the reversible activations (pillar B) AND the counter synapse
//       (pillar A) run together end-to-end on Vulkan (forward under NoGrad +
//       inverse-recovery + recompute + counter state update all through
//       Vulkan device paths). The witness asserts the loss CHANGES across
//       steps (rules out silent skip / no-op backward); a strict "must fall"
//       bound is not robust because the counter is ternary-bounded with
//       stochastic rounding on a tiny budget.
//
// Both witnesses skip cleanly (exit 77) when no Vulkan device is present.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/reversible.hpp>

namespace {

void fill_deterministic(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t s = seed | 1u;
    for (auto& x : v) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x = scale * (static_cast<float>(static_cast<std::int32_t>(s % 2001) - 1000) / 1000.0f);
    }
}

// Witness A: single CounterStateLinear teacher-recovery on Vulkan.
bool witness_counter_recovery(motifcl::Backend& backend) {
    using namespace motifcl;
    const int n = 16, N = 128, C = 11, steps = 200;
    const float teacher_scale = 0.25f, lr = 0.005f;

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> tern(-1, 1);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<int> tw(n * n);
    for (int& w : tw) w = tern(rng);
    std::vector<float> xh(N * n);
    for (float& xi : xh) xi = gauss(rng);
    std::vector<float> yh(N * n, 0.0f);
    for (int r = 0; r < N; ++r)
        for (int o = 0; o < n; ++o) {
            float a = 0.0f;
            for (int i = 0; i < n; ++i) a += xh[r * n + i] * (teacher_scale * (float)tw[o * n + i]);
            yh[r * n + o] = a;
        }

    auto x = Tensor::from_cpu(backend, {N, n}, DType::F32, xh.data());
    auto y = Tensor::from_cpu(backend, {N, n}, DType::F32, yh.data());

    const float base = std::sqrt(3.0f / (2.0f * (float)n));
    nn::CounterStateLinear layer(backend, n, n, C, lr, 0.0f, teacher_scale / base);

    float first_loss = 0.0f, last_loss = 0.0f;
    for (int s = 0; s < steps; ++s) {
        auto pred = layer.forward(x);
        auto diff = sub(pred, y);
        auto diffv = diff.to_vector<float>();
        double acc = 0.0;
        for (float v : diffv) acc += double(v) * double(v);
        const float loss = float(acc / double(N * n));
        if (s == 0) first_loss = loss;
        if (s == steps - 1) last_loss = loss;
        // grad = 2 * diff / (N*n). Materialise on host then upload; backward
        // drives the fused Vulkan counter state update.
        std::vector<float> gh(static_cast<std::size_t>(N * n));
        for (std::size_t i = 0; i < diffv.size(); ++i) gh[i] = 2.0f * diffv[i] / float(N * n);
        auto grad = Tensor::from_cpu(backend, {N, n}, DType::F32, gh.data());
        pred.backward(grad);
    }
    std::cout << "[A counter-recovery] first_loss=" << first_loss << " last_loss=" << last_loss
              << "\n";

    // Decode the final weight and check ternary accuracy vs the teacher.
    auto w = layer.decode_weight().to_vector<float>();
    int correct = 0;
    const float scale_factor = teacher_scale;  // counter stores ternary * scale
    for (std::size_t i = 0; i < w.size(); ++i) {
        const float target = scale_factor * float(tw[i]);
        // decode rounds to ternary * scale; compare with small tolerance.
        if (std::fabs(w[i] - target) <= scale_factor * 0.5f) ++correct;
    }
    const float acc_ratio = float(correct) / float(w.size());
    std::cout << "[A counter-recovery] ternary-acc vs teacher: " << acc_ratio << "\n";

    return last_loss < first_loss * 0.5f && acc_ratio >= 0.80f;
}

// Witness B: 4-block reversible stack with CounterStateLinear coupling.
bool witness_reversible_counter_stack(motifcl::Backend& backend) {
    using namespace motifcl;
    const std::int64_t d = 64;
    const std::int64_t N = 4;
    const int n_blocks = 4;
    const int steps = 50;
    const float lr = 3e-3f;

    std::vector<std::shared_ptr<nn::ReversibleBlock>> stack;
    for (int i = 0; i < n_blocks; ++i) {
        auto f = std::make_shared<nn::CounterStateLinear>(backend, d, d, 11, lr);
        auto g = std::make_shared<nn::CounterStateLinear>(backend, d, d, 11, lr);
        stack.push_back(std::make_shared<nn::ReversibleBlock>(f, g));
    }

    std::vector<float> x1_host(static_cast<std::size_t>(N * d));
    std::vector<float> x2_host(static_cast<std::size_t>(N * d));
    fill_deterministic(x1_host, 0xA1u, 0.4f);
    fill_deterministic(x2_host, 0xA2u, 0.4f);
    auto x1 = Tensor::from_cpu(backend, {N, d}, DType::F32, x1_host.data());
    auto x2 = Tensor::from_cpu(backend, {N, d}, DType::F32, x2_host.data());

    float first_loss = 0.0f, last_loss = 0.0f;
    for (int s = 0; s < steps; ++s) {
        Tensor cur1 = x1, cur2 = x2;
        for (auto& blk : stack) {
            auto [y1, y2] = blk->forward(cur1, cur2);
            cur1 = std::move(y1);
            cur2 = std::move(y2);
        }
        // L = 0.5 * mean(sum_sq(cur)). grad = cur / (2*N*d). Materialise on
        // host then upload; backward drives both the reversible recompute and
        // the counter state updates through Vulkan device paths.
        auto d1 = cur1.to_vector<float>();
        auto d2 = cur2.to_vector<float>();
        double acc = 0.0;
        for (float v : d1) acc += double(v) * double(v);
        for (float v : d2) acc += double(v) * double(v);
        const float loss = float(0.5 * acc / double(2 * N * d));
        if (s == 0) first_loss = loss;
        if (s == steps - 1) last_loss = loss;
        if (s % 10 == 0 || s == steps - 1) std::cout << "[B stack] step " << s << " loss " << loss << "\n";
        std::vector<float> g1_host(static_cast<std::size_t>(N * d));
        std::vector<float> g2_host(static_cast<std::size_t>(N * d));
        for (std::size_t i = 0; i < d1.size(); ++i) {
            g1_host[i] = d1[i] / float(2 * N * d);
            g2_host[i] = d2[i] / float(2 * N * d);
        }
        auto g1 = Tensor::from_cpu(backend, {N, d}, DType::F32, g1_host.data());
        auto g2 = Tensor::from_cpu(backend, {N, d}, DType::F32, g2_host.data());
        cur2.backward(g2);
        cur1.backward(g1);
    }
    std::cout << "[B stack] first_loss=" << first_loss << " last_loss=" << last_loss << "\n";
    // The reversible + counter Vulkan pipeline must run end-to-end AND the
    // loss must CHANGE across steps (rules out silent skip / no-op backward).
    return std::fabs(last_loss - first_loss) > 1.0e-6f;
}

} // namespace

int main() {
    using namespace motifcl;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "Skipping Vulkan memory-native test: "
                  << (probe.error.empty() ? "no Vulkan device" : probe.error) << "\n";
        return 77;
    }

    bool ok = true;
    try {
        Backend backend = Backend::create_vulkan();
        ok = witness_counter_recovery(backend) && ok;
        ok = witness_reversible_counter_stack(backend) && ok;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        ok = false;
    }

    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

