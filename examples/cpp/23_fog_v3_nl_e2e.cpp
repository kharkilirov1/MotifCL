// 23_fog_v3_nl_e2e.cpp
//
// REAL end-to-end natural-language state-tracking test for FOG v3.
//
// Pipeline (model does the NL reading, FOG does the exact reasoning):
//   English story
//     -> frozen lexical backbone  -> hidden at each "." sentence marker
//     -> trained shared loc_head  -> absolute location per sentence (S0..S15)
//     -> deltas[k] = (loc[k] - loc[k-1]) mod 16   (derived, host)
//     -> FOG v3 structured_step loop (the SFT-trained register machine)
//          accumulates  state = (state + delta) % 16  over R steps
//     -> decode_state -> final location.
//
// The model itself reads the named locations from text; the FOG engine executes
// the exact multi-step modular state composition. No human pre-parses the story.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {

constexpr int kStates = 16;
constexpr int kPeriodTok = 17;

const char* const kLocNames[kStates] = {
    "Kitchen","Garden","River","Forest","Treehouse","Lake","Cave",
    "Bridge","Hill","Market","Barn","Field","Orchard","Well","Tower","Cellar"
};
const char* const kOpNames[7] = {"READ","IDENTITY","BLOCK_PRODUCT","FLEX_0","FLEX_1","FLEX_2","FLEX_3"};

std::vector<float> state_code(int s, int d) {
    std::vector<float> out(static_cast<std::size_t>(d));
    const int pairs = d / 2; const float amp = std::sqrt(2.0f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int j = 0; j < pairs; ++j) { const int h = j + 1;
        const float a = 2.0f * kPi * static_cast<float>(s * h) / static_cast<float>(kStates);
        out[2u*j] = amp*std::cos(a); out[2u*j+1u] = amp*std::sin(a); }
    return out;
}
void install_state_codebook(motifcl::nn::FogV3Model& m) {
    auto host = m.lexical.token_embedding.weight.data.to_vector<float>();
    const int V = m.config.vocab_size, d = m.config.d_model;
    for (int s = 0; s < kStates; ++s) { auto c = state_code(s, d); std::copy(c.begin(), c.end(), host.begin() + ptrdiff_t((4+s)*d)); }
    auto& b = m.lexical.token_embedding.weight.data.backend();
    m.lexical.token_embedding.weight.data = motifcl::Tensor::from_cpu(b, {V,d}, motifcl::DType::F32, host.data());
    m.lexical.token_embedding.weight.data.set_requires_grad(false);
    m.lexical.token_embedding.weight.trainable = false;
}
int decode_state(const motifcl::Tensor& v, const motifcl::Tensor& cb, const motifcl::Tensor& nw) {
    auto n = motifcl::rmsnorm(v, nw); auto lg = motifcl::matmul_transpose_b(n, cb);
    auto lh = lg.to_vector<float>();
    int best = 0; float bv = -1e9f;
    for (int s = 0; s < kStates; ++s) if (lh[4+s] > bv) { bv = lh[4+s]; best = s; }
    return best;
}
std::vector<std::string> split_csv(const std::string& s){ std::stringstream ss(s); std::string it; std::vector<std::string> o; while (std::getline(ss,it,',')) o.push_back(it); return o; }
bool read_line(const std::string& l, int& init, std::vector<int>& deltas, std::vector<std::int32_t>& ids) {
    std::stringstream ss(l); std::string dcsv, tcsv;
    ss >> init; if (!ss) return false; ss >> dcsv >> tcsv;
    for (auto& s : split_csv(dcsv)) if (!s.empty()) deltas.push_back(std::atoi(s.c_str()));
    for (auto& s : split_csv(tcsv)) if (!s.empty()) ids.push_back(static_cast<std::int32_t>(std::atoi(s.c_str())));
    return true;
}
std::vector<int> find_markers(const std::vector<std::int32_t>& ids) {
    std::vector<int> m; for (std::size_t i=0;i<ids.size();++i) if (ids[i]==kPeriodTok) m.push_back(static_cast<int>(i)); return m;
}
std::vector<int> derive_states(int init, const std::vector<int>& deltas) {
    std::vector<int> s; s.push_back(init);
    for (int d : deltas) s.push_back((s.back() + d) % kStates);
    return s;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string lexical_ckpt = argc > 1 ? argv[1] : "checkpoints/fog_v3_rx580_lexical_30k.mclp";
        const std::string sft_ckpt      = argc > 2 ? argv[2] : "checkpoints/fog_v3_rx580_reasoning_sft.mclp";
        const std::string binder_ckpt   = argc > 3 ? argv[3] : "checkpoints/fog_v3_rx580_nl_binder.mclp";
        const std::string test_tok      = argc > 4 ? argv[4] : "data/fog_nl_test.tok";
        const std::string story_file    = argc > 5 ? argv[5] : "data/fog_nl_5tests.tok";
        const bool run_5tests = std::filesystem::exists(story_file);

        auto backend = motifcl::Backend::create_vulkan();
        const auto info = backend.device_info();
        motifcl::nn::FogV3Config cfg;
        cfg.vocab_size = 8192; cfg.max_seq_len = 512; cfg.d_model = 320;
        cfg.n_heads = 5; cfg.n_layers = 4; cfg.d_ff = 1344; cfg.dropout = 0.0f;
        motifcl::nn::FogV3Model model(backend, cfg);

        if (!std::filesystem::exists(lexical_ckpt)) { std::cerr << "missing lexical ckpt\n"; return 1; }
        motifcl::load_parameters(model.parameters(), backend, lexical_ckpt);
        if (std::filesystem::exists(sft_ckpt)) { motifcl::load_parameters(model.parameters(), backend, sft_ckpt); }
        install_state_codebook(model);
        for (auto* p : model.parameters()) p->data.set_requires_grad(false);

        motifcl::nn::Linear loc_head(backend, cfg.d_model, kStates, true);
        if (!std::filesystem::exists(binder_ckpt)) { std::cerr << "missing binder ckpt: " << binder_ckpt << "\n"; return 1; }
        motifcl::load_parameters(loc_head.parameters(), backend, binder_ckpt);

        auto norm_w = motifcl::Tensor::ones(backend, {cfg.d_model}, motifcl::DType::F32);
        norm_w.set_requires_grad(false);
        auto codebook = motifcl::rmsnorm(model.lexical.token_embedding.weight.data, norm_w);
        codebook.set_requires_grad(false);
        auto& runtime = backend.vulkan_runtime();

        std::cout << "========================================================================\n";
        std::cout << "  FOG v3 END-TO-END NL TEST (model reads English story, FOG computes)\n";
        std::cout << "  Device: " << info.device_name << "\n";
        std::cout << "========================================================================\n";

        // Predict the absolute location named in each sentence (markers 0..R).
        auto predict_locations = [&](const std::vector<std::int32_t>& ids) -> std::vector<int> {
            motifcl::autograd::NoGradGuard ng;
            auto t = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(ids.size())}, motifcl::DType::I32, ids.data());
            auto h = model.lexical.hidden(t);
            backend.finish();
            auto markers = find_markers(ids);
            std::vector<int> locs;
            for (int m : markers) {
                auto hk = motifcl::slice_rows(h, m, m + 1);
                auto lk_t = loc_head.forward(hk);
                backend.finish();
                auto lhv = lk_t.to_vector<float>();
                locs.push_back(static_cast<int>(std::max_element(lhv.begin(), lhv.end()) - lhv.begin()));
            }
            return locs;  // locs[k] = location read from sentence k
        };

        // --- Part A: measured accuracy on the held-out test set ---
        if (std::filesystem::exists(test_tok)) {
            std::vector<std::string> lines; std::ifstream in(test_tok); std::string l;
            while (std::getline(in,l)) if (!l.empty()) lines.push_back(l);
            const int N = std::min<int>(200, static_cast<int>(lines.size()));
            int sent_ok=0, sent_total=0, full_ok=0, total=0;
            for (int i=0;i<N;++i) {
                int init_gold; std::vector<int> gold_d; std::vector<std::int32_t> ids;
                if (!read_line(lines[i], init_gold, gold_d, ids)) continue;
                int R = static_cast<int>(gold_d.size());
                if (static_cast<int>(find_markers(ids).size()) < R+1) continue;
                auto gold_states = derive_states(init_gold, gold_d);
                auto pred = predict_locations(ids);
                if (static_cast<int>(pred.size()) < R+1) continue;
                bool all = true;
                for (int k=0;k<=R;++k) { if (pred[k]==gold_states[k]) ++sent_ok; else all=false; ++sent_total; }
                if (all) ++full_ok;
                ++total;
            }
            std::cout << "Held-out test (" << total << " stories, " << sent_total << " sentences)\n";
            std::cout << "  location_acc (per sentence) = " << std::fixed << std::setprecision(1)
                      << (100.0f*sent_ok/std::max(1,sent_total)) << "%\n";
            std::cout << "  full_chain (all locs correct) = " << (100.0f*full_ok/std::max(1,total)) << "%\n\n";
        }

        // --- Part B: the user tests in English ---
        if (run_5tests) {
            std::vector<std::string> lines; std::ifstream in(story_file); std::string l;
            while (std::getline(in,l)) if (!l.empty()) lines.push_back(l);
            for (std::size_t t=0;t<lines.size();++t) {
                int init_gold; std::vector<int> gold_d; std::vector<std::int32_t> ids;
                if (!read_line(lines[t], init_gold, gold_d, ids)) continue;
                int R = static_cast<int>(gold_d.size());
                std::cout << "===== USER TEST " << (t+1) << " (R=" << R << ") =====\n";
                auto gold_states = derive_states(init_gold, gold_d);

                auto pred = predict_locations(ids);
                if (static_cast<int>(pred.size()) < R+1) { std::cout << "  (marker count mismatch)\n\n"; continue; }

                std::cout << "  MODEL read locations (from text): ";
                for (int k=0;k<=R;++k){ if(k) std::cout<<" -> "; std::cout<<kLocNames[pred[k]]; }
                std::cout << "\n";

                // derive deltas from predicted locations
                std::vector<int> pred_deltas;
                for (int k=1;k<=R;++k) pred_deltas.push_back((pred[k] - pred[k-1] + kStates) % kStates);
                std::cout << "  derived deltas: {";
                for (int k=0;k<R;++k){ if(k) std::cout<<","; std::cout<<pred_deltas[k]; }
                std::cout << "}   gold deltas: {";
                for (int k=0;k<R;++k){ if(k) std::cout<<","; std::cout<<gold_d[k]; }
                std::cout << "}\n";

                // FOG register machine on predicted (init, deltas)
                int init_pred = pred[0];
                std::int32_t q_tok = 4 + init_pred;
                auto q_t = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &q_tok);
                auto reg = model.initial_state(q_t);
                int decoded_state = init_pred;
                std::vector<int> trace; trace.push_back(init_pred);
                for (int k=0;k<R;++k) {
                    int delta = pred_deltas[k];
                    std::vector<std::int32_t> k_ids(kStates), v_ids(kStates);
                    for (int s=0;s<kStates;++s){ k_ids[s]=4+s; v_ids[s]=4+delta; }
                    auto k_t = motifcl::Tensor::from_cpu(backend,{kStates},motifcl::DType::I32,k_ids.data());
                    auto v_t = motifcl::Tensor::from_cpu(backend,{kStates},motifcl::DType::I32,v_ids.data());
                    const bool batched = runtime.batch_begin();
                    auto o = model.structured_step(reg, k_t, v_t, 1, kStates);
                    reg = o.state;
                    if (batched){ const auto s=runtime.batch_end(); if(!s.success) throw std::runtime_error(s.error);}
                    backend.finish();
                    decoded_state = decode_state(reg.value, codebook, norm_w);
                    trace.push_back(decoded_state);
                }
                std::cout << "  FOG register trace: S";
                for (std::size_t k=0;k<trace.size();++k){ if(k) std::cout<<" -> S"; std::cout<<trace[k]; }
                std::cout << "\n";
                int gold_final = gold_states.back();
                bool ok = (decoded_state == gold_final);
                std::cout << "  MODEL FINAL ANSWER: " << kLocNames[decoded_state] << " (S" << decoded_state << ")\n";
                std::cout << "  GOLD:               " << kLocNames[gold_final] << " (S" << gold_final << ")"
                          << "   verdict=" << (ok ? "CORRECT" : "WRONG") << "\n\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
}