#include <motifcl/nn/group_counter_import.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace motifcl::nn {

namespace {

template <typename T>
void read_exact(std::ifstream& f, T* dst, std::size_t count, const char* what) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(T)));
    if (!f) throw std::runtime_error(std::string("mncc: truncated reading ") + what);
}

std::uint32_t read_u32(std::ifstream& f, const char* what) {
    std::uint32_t v = 0;
    read_exact(f, &v, 1, what);
    return v;
}

} // namespace

std::vector<float> GroupCounterLayer::decode_weight_host() const {
    const int out = out_features, in = in_features;
    std::vector<float> W(static_cast<std::size_t>(out) * in, 0.0f);
    const int n_groups = in / group;
    for (int o = 0; o < out; ++o) {
        const std::int8_t* trow = t.data() + static_cast<std::size_t>(o) * in;
        const float* srow = scale.data() + static_cast<std::size_t>(o) * n_groups;
        float* wrow = W.data() + static_cast<std::size_t>(o) * in;
        for (int j = 0; j < in; ++j) {
            // permuted position j holds original column perm[j]
            wrow[perm[j]] = srow[j / group] * static_cast<float>(trow[j]);
        }
    }
    for (std::size_t k = 0; k < sal_idx.size(); ++k) {
        W[static_cast<std::size_t>(sal_idx[k])] = sal_val[k];
    }
    return W;
}

std::map<std::string, GroupCounterLayer> load_mncc(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("mncc: cannot open " + path);
    char magic[8];
    read_exact(f, magic, 8, "magic");
    if (std::memcmp(magic, "MNCC0001", 8) != 0)
        throw std::runtime_error("mncc: bad magic in " + path);
    const std::uint32_t n_layers = read_u32(f, "n_layers");

    std::map<std::string, GroupCounterLayer> layers;
    for (std::uint32_t li = 0; li < n_layers; ++li) {
        GroupCounterLayer L;
        const std::uint32_t plen = read_u32(f, "path_len");
        L.path.resize(plen);
        read_exact(f, L.path.data(), plen, "path");
        L.out_features = static_cast<int>(read_u32(f, "out"));
        L.in_features = static_cast<int>(read_u32(f, "in"));
        L.group = static_cast<int>(read_u32(f, "group"));
        L.C = static_cast<int>(read_u32(f, "C"));
        if (L.group <= 0 || L.in_features % L.group != 0)
            throw std::runtime_error("mncc: bad group for " + L.path);
        const std::size_t n = static_cast<std::size_t>(L.out_features) * L.in_features;
        L.perm.resize(L.in_features);
        read_exact(f, L.perm.data(), L.perm.size(), "perm");
        L.t.resize(n);
        read_exact(f, L.t.data(), n, "t");
        L.c.resize(n);
        read_exact(f, L.c.data(), n, "c");
        L.scale.resize(static_cast<std::size_t>(L.out_features) * (L.in_features / L.group));
        read_exact(f, L.scale.data(), L.scale.size(), "scale");
        const std::uint32_t n_sal = read_u32(f, "n_salient");
        L.sal_idx.resize(n_sal);
        read_exact(f, L.sal_idx.data(), n_sal, "sal_idx");
        L.sal_val.resize(n_sal);
        read_exact(f, L.sal_val.data(), n_sal, "sal_val");
        layers.emplace(L.path, std::move(L));
    }
    return layers;
}

} // namespace motifcl::nn
