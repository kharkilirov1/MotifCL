// Bridge witness: an MNCC0001 container exported by memory-native-training
// (scripts/export_motifcl.py) must decode BIT-EXACTLY to the PyTorch reference
// visible_weight() captured next to it (tests/data/mncc_case/ref_weight.bin).
// This test is pure host-side: it pins the container format and the decode rule
// (group scales on the permuted axis + salient overrides) before any GPU work.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include <motifcl/nn/group_counter_import.hpp>

#include "test_utils.hpp"

int main(int argc, char** argv) {
    try {
        std::string dir = "tests/data/mncc_case";
        if (argc > 1) dir = argv[1];
        for (const char* candidate : {"", "../", "../../"}) {
            std::ifstream probe(std::string(candidate) + dir + "/counters.mncc",
                                std::ios::binary);
            if (probe) { dir = std::string(candidate) + dir; break; }
        }

        auto layers = motifcl::nn::load_mncc(dir + "/counters.mncc");
        if (layers.size() != 1) {
            std::cerr << "expected 1 layer, got " << layers.size() << "\n";
            return 1;
        }
        const auto& L = layers.begin()->second;
        std::printf("layer %s: out=%d in=%d group=%d C=%d salient=%zu\n",
                    L.path.c_str(), L.out_features, L.in_features, L.group, L.C,
                    L.sal_idx.size());
        if (L.sal_idx.empty()) {
            std::cerr << "test vector must carry a salient channel\n";
            return 1;
        }

        std::ifstream rf(dir + "/ref_weight.bin", std::ios::binary);
        if (!rf) { std::cerr << "missing ref_weight.bin\n"; return 1; }
        std::uint32_t ro = 0, ri = 0;
        rf.read(reinterpret_cast<char*>(&ro), 4);
        rf.read(reinterpret_cast<char*>(&ri), 4);
        std::vector<float> ref(static_cast<std::size_t>(ro) * ri);
        rf.read(reinterpret_cast<char*>(ref.data()),
                static_cast<std::streamsize>(ref.size() * sizeof(float)));
        if (!rf) { std::cerr << "truncated ref_weight.bin\n"; return 1; }
        if ((int)ro != L.out_features || (int)ri != L.in_features) {
            std::cerr << "ref shape mismatch\n";
            return 1;
        }

        const std::vector<float> W = L.decode_weight_host();
        double max_abs = 0.0;
        std::size_t bad = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double d = std::fabs((double)W[i] - (double)ref[i]);
            if (d > max_abs) max_abs = d;
            if (d > 1e-6) ++bad;
        }
        std::printf("decode parity: max|diff|=%.3g bad=%zu of %zu\n",
                    max_abs, bad, ref.size());
        if (bad != 0) {
            std::cerr << "FAIL: decode does not match the PyTorch reference\n";
            return 1;
        }
        std::puts("PASS test_group_counter_import");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
}
