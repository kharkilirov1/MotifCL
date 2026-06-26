// Witness for reversible viability follow-up: is the measured 3e-3 reconstruction error
// tolerable for training? Train a char-LM MLP clean vs with ~3e-3 noise injected into the
// hidden activations each step (imitating recompute error in a reversible backward).
// Pass = noisy final ce within a few percent of clean.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/embedding.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/loss.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

using namespace motifcl;

static std::vector<unsigned char> read_bytes(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in.good()) return {};
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1]
        : "C:/Users/Kharki/AppData/Local/Temp/claude/C--Users-Kharki-BitNet/00788cf4-aecf-4c90-9b96-a0b06c03cad6/scratchpad/counter_research/tinyshakespeare.txt";
    auto bytes = read_bytes(path);
    if (bytes.size() < 10000) { printf("SKIP: dataset not found\n"); return 0; }

    auto backend = Backend::create_opencl();
    const int vocab = 128, N = 4096, embd = 64, hid = 256, steps = 400;
    std::mt19937 rng(0);
    std::uniform_int_distribution<std::size_t> pos(0, bytes.size() - 2);
    std::vector<std::int32_t> xi(N), yi(N);
    for (int k = 0; k < N; ++k) { std::size_t p = pos(rng); xi[k] = bytes[p] % vocab; yi[k] = bytes[p + 1] % vocab; }
    auto x = Tensor::from_cpu(backend, {N}, DType::I32, xi.data());
    auto y = Tensor::from_cpu(backend, {N}, DType::I32, yi.data());

    auto run = [&](float noise_std) -> float {
        nn::Embedding emb(backend, vocab, embd);
        nn::Linear l1(backend, embd, hid, false), l2(backend, hid, vocab, false);
        std::vector<nn::Parameter*> ps;
        for (auto* m : {(nn::Module*)&emb, (nn::Module*)&l1, (nn::Module*)&l2})
            for (auto* p : m->parameters()) ps.push_back(p);
        optim::Adam opt(ps, 1e-3f);
        float last = 0.0f;
        for (int s = 0; s < steps; ++s) {
            auto h = relu(l1.forward(emb.forward(x)));
            if (noise_std > 0.0f) h = add(h, Tensor::randn(backend, h.shape(), noise_std));
            auto logits = l2.forward(h);
            auto loss = softmax_cross_entropy(logits, y);
            opt.zero_grad();
            loss.backward();
            opt.step();
            if (s == steps - 1) last = loss.item();
        }
        return last;
    };

    float clean = run(0.0f);
    float noisy = run(3e-3f);
    printf("char-LM MLP, %d steps:\n", steps);
    printf("  clean         final ce = %.4f\n", clean);
    printf("  +3e-3 noise   final ce = %.4f\n", noisy);
    printf("  delta = %.4f (%.1f%%)\n", noisy - clean, (noisy - clean) / clean * 100.0f);
    const bool ok = noisy < clean * 1.05f + 1e-4f;
    printf(ok ? "PASS: 3e-3 reconstruction error is tolerable for training\n"
              : "REFUTED: 3e-3 hurts training -> fixed-point/anchor needed\n");
    return ok ? 0 : 1;
}
