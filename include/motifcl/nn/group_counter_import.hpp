#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <motifcl/tensor/tensor.hpp>

namespace motifcl::nn {

// Imported memory-native deploy state: the group-128 act-ordered ternary format
// produced by the PyTorch solver (memory-native-training repo) and shipped as an
// ``MNCC0001`` container by scripts/export_motifcl.py.
//
// Per layer the container carries, in PERMUTED column order:
//   t     i8  [out, in]      ternary codes {-1, 0, +1}
//   c     i8  [out, in]      residual counters (alpha=0 inference ignores them)
//   scale f32 [out, in/group] per-(row, group) symmetric scales
//   perm  i32 [in]           permuted position j holds ORIGINAL column perm[j]
//   salient (idx i64 into the flat ORIGINAL-order [out, in], val f32) exact overrides
//
// Dense reconstruction (must match PackedGroupScaleCounterLinear.visible_weight()
// at alpha=0, bit-exact in f32):
//   W[o, perm[j]] = scale[o, j / group] * t[o, j]
//   W.flat[sal_idx[k]] = sal_val[k]
//
// This class is the INFERENCE bridge: decode + forward. Training the imported
// group format natively (group-scale counter update kernels) is a follow-up.
struct GroupCounterLayer {
    std::string path;
    int out_features = 0;
    int in_features = 0;
    int group = 0;
    int C = 11;
    std::vector<std::int32_t> perm;      // [in]
    std::vector<std::int8_t> t;          // [out*in] permuted order
    std::vector<std::int8_t> c;          // [out*in] permuted order
    std::vector<float> scale;            // [out * (in/group)]
    std::vector<std::int64_t> sal_idx;   // flat original-order indices
    std::vector<float> sal_val;

    // Dense f32 [out, in] weight on the host, alpha=0 (t channel + salient only).
    std::vector<float> decode_weight_host() const;
};

// Parses an MNCC0001 container. Throws std::runtime_error on malformed input.
std::map<std::string, GroupCounterLayer> load_mncc(const std::string& path);

} // namespace motifcl::nn
