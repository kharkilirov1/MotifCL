#pragma once

#include <cstdint>
#include <vector>

namespace motifcl::solver {

// Phase-1 host-side port of the memory-native PTQ solver core
// (memory-native-training/src/memory_native/donor/ptq.py). Numerics follow the
// PyTorch reference exactly; parity is pinned by tests/test_ternary_solver.cpp
// against vectors exported from the reference implementation.
//
// Roadmap (docs/MN_SOLVER_PORT_PLAN.md): act-order GPTQ group sweep with error
// feedback -> exact per-row group-scale align (Cholesky) -> itf asymmetric grid ->
// salient split -> asymmetric two-tower calibration. This file is the first rung:
// the exact data-free row minimizer and the H-diagonal-weighted group scale refit.

// Exact per-row L2 minimizer of ||w - s*t|| over t in {-1,0,+1}, s >= 1e-8.
// w is row-major [out, in]; outputs: s [out], t [out*in] in {-1,0,+1}.
void optimal_ternary(const float* w, int out_features, int in_features,
                     std::vector<float>& s, std::vector<std::int8_t>& t);

// Per-(row, group) scale refit on a FIXED ternary support, weighted by the
// Hessian diagonal (the "hdiag" refit of the v3 solver):
//   s[o, g] = sum_j hdiag[j] * w[o,j] * t[o,j] / max(sum_j hdiag[j] * t[o,j]^2, eps)
// with j running over group g's columns. Groups live on the given column order.
void group_scale_refit_hdiag(const float* w, const std::int8_t* t,
                             const float* hdiag, int out_features, int in_features,
                             int group, std::vector<float>& scales,
                             float eps = 1e-12f);

} // namespace motifcl::solver
