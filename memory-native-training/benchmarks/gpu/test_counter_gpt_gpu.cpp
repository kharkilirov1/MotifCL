// Counter layer in a FULL attention transformer block on real char-LM with context.
// One GPT block (counter q/k/v/o + counter MLP, self-updating) vs the same block with
// dense FP32 Linear + Adam, on tinyshakespeare sequences. Expect counter parity-class.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/embedding.hpp>
#include <motifcl/nn/rmsnorm.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/loss.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
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
    const int vocab = 128, batch = 16, seq = 32, n_embd = 64, n_head = 4, steps = 300;
    const int tokens = batch * seq;

    std::mt19937 rng(0);
    std::uniform_int_distribution<std::size_t> start(0, bytes.size() - seq - 2);
    std::vector<std::int32_t> xi(tokens), yi(tokens), pi(tokens);
    for (int b = 0; b < batch; ++b) {
        std::size_t s = start(rng);
        for (int t = 0; t < seq; ++t) {
            xi[b * seq + t] = (std::int32_t)(bytes[s + t] % vocab);
            yi[b * seq + t] = (std::int32_t)(bytes[s + t + 1] % vocab);
            pi[b * seq + t] = t;
        }
    }
    auto x = Tensor::from_cpu(backend, {tokens}, DType::I32, xi.data());
    auto y = Tensor::from_cpu(backend, {tokens}, DType::I32, yi.data());
    auto pos = Tensor::from_cpu(backend, {tokens}, DType::I32, pi.data());

    auto run = [&](bool counter) -> float {
        nn::Embedding tok(backend, vocab, n_embd), pemb(backend, n_embd, n_embd);
        nn::Embedding pos_emb(backend, seq, n_embd);
        nn::RMSNorm rms1(backend, n_embd), rms2(backend, n_embd), rmsf(backend, n_embd);
        nn::Linear head(backend, n_embd, vocab, false);
        // counter or dense projections
        auto mk = [&](int in, int out) {
            return std::make_shared<nn::CounterStateLinear>(backend, in, out, 11, 0.004f, 2e-3f);
        };
        auto mkd = [&](int in, int out) { return std::make_shared<nn::Linear>(backend, in, out, false); };
        std::shared_ptr<nn::Module> cq, ck, cv, co, cfc, cfc2;
        if (counter) { cq = mk(n_embd, n_embd); ck = mk(n_embd, n_embd); cv = mk(n_embd, n_embd);
                       co = mk(n_embd, n_embd); cfc = mk(n_embd, 4 * n_embd); cfc2 = mk(4 * n_embd, n_embd); }
        else { cq = mkd(n_embd, n_embd); ck = mkd(n_embd, n_embd); cv = mkd(n_embd, n_embd);
               co = mkd(n_embd, n_embd); cfc = mkd(n_embd, 4 * n_embd); cfc2 = mkd(4 * n_embd, n_embd); }

        // Adam over the dense/non-counter params (embeddings, norms, head; + proj if dense)
        std::vector<nn::Parameter*> ap;
        for (auto* m : {(nn::Module*)&tok, (nn::Module*)&pos_emb, (nn::Module*)&rms1,
                        (nn::Module*)&rms2, (nn::Module*)&rmsf, (nn::Module*)&head})
            for (auto* p : m->parameters()) ap.push_back(p);
        if (!counter)
            for (auto* m : {cq.get(), ck.get(), cv.get(), co.get(), cfc.get(), cfc2.get()})
                for (auto* p : m->parameters()) ap.push_back(p);
        optim::Adam opt(ap, 1e-3f);

        float last = 0.0f;
        for (int s = 0; s < steps; ++s) {
            auto emb = add(tok.forward(x), pos_emb.forward(pos));
            auto h = rms1.forward(emb);
            auto ctx = multihead_attention(cq->forward(h), ck->forward(h), cv->forward(h),
                                           n_head, true, batch, seq);
            auto x1 = add(emb, co->forward(ctx));
            auto h2 = rms2.forward(x1);
            auto x2 = add(x1, cfc2->forward(gelu(cfc->forward(h2))));
            auto logits = head.forward(rmsf.forward(x2));
            auto loss = softmax_cross_entropy(logits, y);
            opt.zero_grad();
            loss.backward();
            opt.step();
            if (s % 60 == 0) printf("  %s step=%3d ce=%.4f\n", counter ? "counter" : "dense  ", s, loss.item());
            if (s == steps - 1) last = loss.item();
        }
        return last;
    };

    float c = run(true);
    float d = run(false);
    printf("counter-GPT final ce=%.4f   dense-GPT final ce=%.4f   counter/dense=%.2fx\n", c, d, c / (d + 1e-9f));
    const bool ok = c < 1.4f * d;
    printf(ok ? "PASS: native counter works in a full attention block, parity-class\n"
              : "CHECK: outside band\n");
    return ok ? 0 : 1;
}
