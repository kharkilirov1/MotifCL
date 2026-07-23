#include <motifcl/solver/ternary_solver.hpp>

#include <algorithm>
#include <cmath>

namespace motifcl::solver {

void optimal_ternary(const float* w, int out_features, int in_features,
                     std::vector<float>& s, std::vector<std::int8_t>& t) {
    s.assign(out_features, 0.0f);
    t.assign(static_cast<std::size_t>(out_features) * in_features, 0);
    std::vector<float> absw(in_features);
    for (int o = 0; o < out_features; ++o) {
        const float* wrow = w + static_cast<std::size_t>(o) * in_features;
        for (int j = 0; j < in_features; ++j) absw[j] = std::fabs(wrow[j]);
        std::sort(absw.begin(), absw.end(), std::greater<float>());
        // argmax_k (prefix_sum_k)^2 / k  (k is 1-based), exactly like the reference:
        // ties resolve to the FIRST maximizer (torch.argmax picks the first).
        double csum = 0.0;
        double best = -1.0;
        int kstar = 0;           // 0-based index of the winning k
        double best_csum = 0.0;
        for (int k = 0; k < in_features; ++k) {
            csum += absw[k];
            const double obj = csum * csum / (double)(k + 1);
            if (obj > best) {
                best = obj;
                kstar = k;
                best_csum = csum;
            }
        }
        float scale = (float)(best_csum / (double)(kstar + 1));
        if (scale < 1e-8f) scale = 1e-8f;
        const float thr = absw[kstar];
        s[o] = scale;
        std::int8_t* trow = t.data() + static_cast<std::size_t>(o) * in_features;
        for (int j = 0; j < in_features; ++j) {
            if (std::fabs(wrow[j]) >= thr && wrow[j] != 0.0f)
                trow[j] = wrow[j] > 0.0f ? 1 : -1;
        }
    }
}

void group_scale_refit_hdiag(const float* w, const std::int8_t* t,
                             const float* hdiag, int out_features, int in_features,
                             int group, std::vector<float>& scales, float eps) {
    const int n_groups = in_features / group;
    scales.assign(static_cast<std::size_t>(out_features) * n_groups, 0.0f);
    for (int o = 0; o < out_features; ++o) {
        const float* wrow = w + static_cast<std::size_t>(o) * in_features;
        const std::int8_t* trow = t + static_cast<std::size_t>(o) * in_features;
        for (int g = 0; g < n_groups; ++g) {
            double num = 0.0, den = 0.0;
            for (int j = g * group; j < (g + 1) * group; ++j) {
                const double tv = (double)trow[j];
                num += (double)hdiag[j] * (double)wrow[j] * tv;
                den += (double)hdiag[j] * tv * tv;
            }
            if (den < eps) den = eps;
            scales[static_cast<std::size_t>(o) * n_groups + g] = (float)(num / den);
        }
    }
}

} // namespace motifcl::solver
