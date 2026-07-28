// Solver-core parity witness: the C++ optimal_ternary and hdiag group refit must
// reproduce the PyTorch reference (memory-native-training donor/ptq.py) on the
// exported vectors in tests/data/solver_case/case.bin.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include <motifcl/solver/ternary_solver.hpp>

int main(int argc, char** argv) {
    try {
        std::string dir = "tests/data/solver_case";
        if (argc > 1) dir = argv[1];
        std::ifstream f;
        for (const char* candidate : {"", "../", "../../"}) {
            f.open(std::string(candidate) + dir + "/case.bin", std::ios::binary);
            if (f) { dir = std::string(candidate) + dir; break; }
            f.clear();
        }
        if (!f) { std::cerr << "missing case.bin\n"; return 1; }

        std::uint32_t out_f = 0, in_f = 0, group = 0;
        f.read(reinterpret_cast<char*>(&out_f), 4);
        f.read(reinterpret_cast<char*>(&in_f), 4);
        f.read(reinterpret_cast<char*>(&group), 4);
        const std::size_t n = static_cast<std::size_t>(out_f) * in_f;
        std::vector<float> W(n), hdiag(in_f), s_ref(out_f);
        std::vector<std::int8_t> t_ref(n);
        std::vector<float> gs_ref(static_cast<std::size_t>(out_f) * (in_f / group));
        f.read(reinterpret_cast<char*>(W.data()), (std::streamsize)(n * 4));
        f.read(reinterpret_cast<char*>(hdiag.data()), (std::streamsize)(in_f * 4));
        f.read(reinterpret_cast<char*>(s_ref.data()), (std::streamsize)(out_f * 4));
        f.read(reinterpret_cast<char*>(t_ref.data()), (std::streamsize)n);
        f.read(reinterpret_cast<char*>(gs_ref.data()),
               (std::streamsize)(gs_ref.size() * 4));
        if (!f) { std::cerr << "truncated case.bin\n"; return 1; }

        std::vector<float> s;
        std::vector<std::int8_t> t;
        motifcl::solver::optimal_ternary(W.data(), (int)out_f, (int)in_f, s, t);

        double s_max = 0.0;
        for (std::size_t o = 0; o < out_f; ++o)
            s_max = std::max(s_max, (double)std::fabs(s[o] - s_ref[o]));
        std::size_t t_bad = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (t[i] != t_ref[i]) ++t_bad;
        std::printf("optimal_ternary: max|s-s_ref|=%.3g  code mismatches=%zu of %zu\n",
                    s_max, t_bad, n);

        std::vector<float> gs;
        motifcl::solver::group_scale_refit_hdiag(W.data(), t_ref.data(), hdiag.data(),
                                                 (int)out_f, (int)in_f, (int)group, gs);
        double g_max = 0.0;
        for (std::size_t i = 0; i < gs.size(); ++i)
            g_max = std::max(g_max, (double)std::fabs(gs[i] - gs_ref[i]));
        std::printf("hdiag refit: max|s-s_ref|=%.3g over %zu scales\n", g_max, gs.size());

        if (t_bad != 0 || s_max > 5e-6 || g_max > 5e-6) {
            std::cerr << "FAIL: solver core does not match the PyTorch reference\n";
            return 1;
        }
        std::puts("PASS test_ternary_solver");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
}
