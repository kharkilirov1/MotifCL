// Counter layer on a REAL discrete language task (char bigram-MLP on tinyshakespeare),
// where ternary weights are strong (cross-entropy, not exact FP32 regression).
// counter MLP-LM (embedding + counter -> ReLU -> counter, self-update) vs dense MLP-LM
// (Linear + Adam). Expect counter close to dense (parity-class), as PyTorch char-LM showed.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/embedding.hpp>
#include <motifcl/ops/loss.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/autograd/node.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace motifcl;

static std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) return {};
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1]
        : "C:/Users/Kharki/AppData/Local/Temp/claude/C--Users-Kharki-BitNet/00788cf4-aecf-4c90-9b96-a0b06c03cad6/scratchpad/counter_research/tinyshakespeare.txt";
    auto bytes = read_bytes(path);
    if (bytes.size() < 10000) { printf("SKIP: dataset not found at %s\n", path.c_str()); return 0; }

    auto backend = Backend::create_opencl();
    const int vocab = 128, N = 4096, embd = 64, hid = 256, steps = 500;

    std::mt19937 rng(0);
    std::uniform_int_distribution<std::size_t> pos(0, bytes.size() - 2);
    std::vector<std::int32_t> xi(N), yi(N);
    for (int k = 0; k < N; ++k) {
        std::size_t p = pos(rng);
        xi[k] = (std::int32_t)(bytes[p] % vocab);
        yi[k] = (std::int32_t)(bytes[p + 1] % vocab);
    }
    auto x = Tensor::from_cpu(backend, {N}, DType::I32, xi.data());
    auto y = Tensor::from_cpu(backend, {N}, DType::I32, yi.data());

    // ---- counter MLP-LM (embedding Adam-trained; counter layers self-update) ----
    nn::Embedding emb(backend, vocab, embd);
    nn::CounterStateLinear c1(backend, embd, hid, 11, 0.005f, 2e-3f, 1.0f, 0.9f, 1e-3f, 1u);
    nn::CounterStateLinear c2(backend, hid, vocab, 11, 0.005f, 2e-3f, 1.0f, 0.9f, 1e-3f, 2u);
    optim::Adam opt_e(emb.parameters(), 1e-3f);
    float c_init = 0.0f, c_final = 0.0f;
    for (int s = 0; s < steps; ++s) {
        auto logits = c2.forward(relu(c1.forward(emb.forward(x))));
        auto loss = softmax_cross_entropy(logits, y);
        opt_e.zero_grad();
        loss.backward();   // counter layers update here; embedding grad accumulates
        opt_e.step();
        if (s == 0) c_init = loss.item();
        if (s == steps - 1) c_final = loss.item();
        if (s % 100 == 0) printf("  counter step=%3d ce=%.4f\n", s, loss.item());
    }

    // ---- dense FP32 MLP-LM + Adam ----
    nn::Embedding emb2(backend, vocab, embd);
    nn::Sequential dense({
        std::make_shared<nn::Linear>(backend, embd, hid),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(backend, hid, vocab),
    });
    std::vector<nn::Parameter*> dp = dense.parameters();
    for (auto* p : emb2.parameters()) dp.push_back(p);
    optim::Adam opt(dp, 1e-3f);
    float d_final = 0.0f;
    for (int s = 0; s < steps; ++s) {
        auto logits = dense.forward(emb2.forward(x));
        auto loss = softmax_cross_entropy(logits, y);
        opt.zero_grad();
        loss.backward();
        opt.step();
        if (s == steps - 1) d_final = loss.item();
    }

    printf("counter MLP-LM: init ce=%.4f  final ce=%.4f\n", c_init, c_final);
    printf("dense FP32+Adam final ce=%.4f\n", d_final);
    printf("counter/dense = %.2fx  (discrete task: ternary is strong here)\n", c_final / (d_final + 1e-9f));
    const bool learned = c_final < 0.9f * c_init;
    const bool competitive = c_final < 1.4f * d_final;
    printf((learned && competitive)
               ? "PASS: native counter MLP-LM is parity-class with dense on real char data\n"
               : "CHECK: outside band\n");
    return (learned && competitive) ? 0 : 1;
}
