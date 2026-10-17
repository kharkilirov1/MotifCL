// 22_fog_v3_nl_binder_train.cpp
//
// Train the MISSING NL binding layer for FOG v3.
//
// Architecture (the fix that makes delta learnable):
//   One SHARED location head (Linear d_model -> 16) that predicts the absolute
//   location named in EACH sentence (sentence 0..R). Reading a location word
//   from a sentence ending is linearly decodable from the frozen backbone's
//   hidden state (proven: init_acc 99.7% after 1 epoch). The per-step "delta"
//   is then derived trivially as (loc[k] - loc[k-1]) mod 16, and the FOG
//   register machine performs the exact multi-step state accumulation.
//
//   Predicting delta directly from hidden[k] fails (a single Linear cannot
//   compute a mod-16 difference of two separately-encoded locations).
//
// Output: checkpoints/fog_v3_rx580_nl_binder.mclp  (loc_head tensors)
// Data:   data/fog_nl_train.tok   "<init> <deltas_csv> <token_id_csv>"

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

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out; std::stringstream ss(s); std::string it;
    while (std::getline(ss, it, ',')) out.push_back(it);
    return out;
}

std::vector<int> find_markers(const std::vector<std::int32_t>& ids) {
    std::vector<int> m;
    for (std::size_t i = 0; i < ids.size(); ++i) if (ids[i] == kPeriodTok) m.push_back(static_cast<int>(i));
    return m;
}

// derive the visited-state sequence from (init, deltas)
std::vector<int> derive_states(int init, const std::vector<int>& deltas) {
    std::vector<int> s; s.push_back(init);
    for (int d : deltas) s.push_back((s.back() + d) % kStates);
    return s;
}

std::size_t param_count(const std::vector<motifcl::nn::Parameter*>& ps) {
    std::size_t n = 0;
    for (auto* p : ps) if (p && p->data.valid()) n += static_cast<std::size_t>(p->data.numel());
    return n;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string lexical_ckpt = argc > 1 ? argv[1] : "checkpoints/fog_v3_rx580_lexical_30k.mclp";
        const std::string train_tok    = argc > 2 ? argv[2] : "data/fog_nl_train.tok";
        const std::string test_tok     = argc > 3 ? argv[3] : "data/fog_nl_test.tok";
        const std::string out_ckpt     = argc > 4 ? argv[4] : "checkpoints/fog_v3_rx580_nl_binder.mclp";
        const int epochs   = argc > 5 ? std::max(1, std::atoi(argv[5])) : 6;
        const float lr     = argc > 6 ? std::max(1e-5f, std::strtof(argv[6], nullptr)) : 3e-3f;

        if (!std::filesystem::exists(train_tok) || !std::filesystem::exists(test_tok)) {
            std::cerr << "Missing pre-tokenized data.\n"; return 1;
        }

        auto backend = motifcl::Backend::create_vulkan();
        const auto info = backend.device_info();

        motifcl::nn::FogV3Config cfg;
        cfg.vocab_size = 8192; cfg.max_seq_len = 512; cfg.d_model = 320;
        cfg.n_heads = 5; cfg.n_layers = 4; cfg.d_ff = 1344; cfg.dropout = 0.0f;
        motifcl::nn::FogV3Model model(backend, cfg);

        if (std::filesystem::exists(lexical_ckpt)) {
            motifcl::load_parameters(model.parameters(), backend, lexical_ckpt);
            std::cout << "Loaded lexical backbone: " << lexical_ckpt << "\n";
        }
        for (auto* p : model.parameters()) { p->trainable = false; p->data.set_requires_grad(false); }

        // Single shared location head.
        motifcl::nn::Linear loc_head(backend, cfg.d_model, kStates, true);
        std::vector<motifcl::nn::Parameter*> head_params = loc_head.parameters();
        for (auto* p : head_params) p->trainable = true;

        std::cout << "FOG v3 NL-binder training (shared location head)\n"
                  << "Device: " << info.device_name << "\n"
                  << "Trainable params: " << param_count(head_params) << "\n"
                  << "LR: " << lr << "  Epochs: " << epochs << "\n";

        motifcl::optim::Adam opt(head_params, lr, 0.9f, 0.95f, 1e-8f, 0.0f);

        // Un-batched hidden: slice_rows() downloads its input on the host, which
        // is illegal while a Vulkan command batch is open. No batch => legal.
        auto run_hidden = [&](const std::vector<std::int32_t>& ids) -> motifcl::Tensor {
            auto t = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(ids.size())},
                                               motifcl::DType::I32, ids.data());
            auto h = model.lexical.hidden(t);
            backend.finish();
            return h;
        };

        auto read_lines = [](const std::string& path){ std::vector<std::string> v; std::ifstream in(path); std::string l; while (std::getline(in,l)) if (!l.empty()) v.push_back(l); return v; };
        auto train_lines = read_lines(train_tok);
        auto test_lines  = read_lines(test_tok);
        std::cout << "train stories: " << train_lines.size() << "  test stories: " << test_lines.size() << "\n\n";

        std::mt19937 rng(7u);

        // accuracy = per-sentence location correct + full-chain (all locations correct -> deltas derived correct)
        auto eval_acc = [&](const std::vector<std::string>& lines) -> std::pair<float,float> {
            motifcl::autograd::NoGradGuard ng;
            int sent_ok = 0, sent_total = 0, full_ok = 0, total = 0;
            for (const auto& l : lines) {
                std::stringstream ss(l);
                int init_lab; ss >> init_lab; std::string dcsv, tcsv; ss >> dcsv >> tcsv;
                auto dv = split_csv(dcsv);
                std::vector<int> deltas; deltas.reserve(dv.size());
                for (auto& s : dv) if (!s.empty()) deltas.push_back(std::atoi(s.c_str()));
                std::vector<std::int32_t> ids;
                for (auto& s : split_csv(tcsv)) if (!s.empty()) ids.push_back(static_cast<std::int32_t>(std::atoi(s.c_str())));
                int R = static_cast<int>(deltas.size());
                auto markers = find_markers(ids);
                if (static_cast<int>(markers.size()) < R + 1) continue;
                auto states = derive_states(init_lab, deltas);
                auto h = run_hidden(ids);
                std::vector<int> pred;
                bool all = true;
                for (int k = 0; k <= R; ++k) {
                    auto hk = motifcl::slice_rows(h, markers[k], markers[k] + 1);
                    auto lk_t = loc_head.forward(hk);
                    backend.finish();
                    auto lhv = lk_t.to_vector<float>();
                    int p = static_cast<int>(std::max_element(lhv.begin(), lhv.end()) - lhv.begin());
                    pred.push_back(p);
                    if (p == states[k]) ++sent_ok; else all = false;
                    ++sent_total;
                }
                if (all) ++full_ok;
                ++total;
            }
            return {static_cast<float>(sent_ok)/std::max(1,sent_total),
                    static_cast<float>(full_ok)/std::max(1,total)};
        };

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::cout << "=== Epoch " << epoch << "/" << epochs << " ===\n";
            std::vector<int> idx(train_lines.size());
            for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
            std::shuffle(idx.begin(), idx.end(), rng);

            double running = 0.0; int shown = 0;
            const int minibatch = 16;
            opt.zero_grad();
            int acc = 0;
            for (int gi = 0; gi < static_cast<int>(idx.size()); ++gi) {
                const auto& l = train_lines[static_cast<std::size_t>(idx[static_cast<std::size_t>(gi)])];
                std::stringstream ss(l);
                int init_lab; ss >> init_lab; std::string dcsv, tcsv; ss >> dcsv >> tcsv;
                auto dv = split_csv(dcsv);
                std::vector<int> deltas; deltas.reserve(dv.size());
                for (auto& s : dv) if (!s.empty()) deltas.push_back(std::atoi(s.c_str()));
                std::vector<std::int32_t> ids;
                for (auto& s : split_csv(tcsv)) if (!s.empty()) ids.push_back(static_cast<std::int32_t>(std::atoi(s.c_str())));
                int R = static_cast<int>(deltas.size());
                auto markers = find_markers(ids);
                if (static_cast<int>(markers.size()) < R + 1) continue;
                auto states = derive_states(init_lab, deltas);

                const auto t0 = std::chrono::steady_clock::now();
                auto h = run_hidden(ids);
                // one cross-entropy per sentence marker, all into the shared head
                for (int k = 0; k <= R; ++k) {
                    auto hk = motifcl::slice_rows(h, markers[k], markers[k] + 1);
                    auto lk = loc_head.forward(hk);
                    std::int32_t lab = static_cast<std::int32_t>(states[k]);
                    auto lt = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &lab);
                    auto loss = motifcl::softmax_cross_entropy(lk, lt);
                    auto ws = motifcl::scale(loss, 1.0f / static_cast<float>(R + 1));
                    ws.backward();
                }
                backend.finish();
                const auto t1 = std::chrono::steady_clock::now();
                running += std::chrono::duration<double, std::milli>(t1 - t0).count();

                ++acc;
                if (acc % minibatch == 0) { opt.step(); opt.zero_grad(); }
                ++shown;
                if (shown % 1000 == 0) {
                    std::cout << "  step " << std::setw(5) << shown << "/" << train_lines.size()
                              << " avg " << std::fixed << std::setprecision(1) << running / shown << "ms/ex\n" << std::flush;
                }
            }
            if (acc % minibatch != 0) { opt.step(); opt.zero_grad(); }

            auto [sent_acc, full_acc] = eval_acc(test_lines);
            std::cout << "  epoch " << epoch << " test | loc_acc(sent)=" << std::fixed
                      << std::setprecision(2) << (sent_acc * 100) << "%"
                      << " full_chain=" << (full_acc * 100) << "%\n\n";
        }

        std::filesystem::create_directories(std::filesystem::path(out_ckpt).parent_path());
        motifcl::save_parameters(head_params, out_ckpt);
        std::cout << "Saved NL-binder loc_head -> " << out_ckpt << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
}