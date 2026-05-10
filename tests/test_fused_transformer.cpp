#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

namespace {

bool close_vec(const std::vector<float>& a, const std::vector<float>& b, float tol, const char* name) {
    if (a.size() != b.size()) {
        std::cerr << name << " size mismatch\n";
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) {
            std::cerr << name << " mismatch at " << i << ": got " << a[i]
                      << " expected " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

motifcl::Tensor make_tensor(motifcl::Backend& backend, const motifcl::Shape& shape, std::vector<float> values) {
    return motifcl::Tensor::from_cpu(backend, shape, motifcl::DType::F32, values.data());
}

void set_fused_env(const char* value) {
#ifdef _WIN32
    _putenv_s("MOTIFCL_ENABLE_FUSED_MLP_BACKWARD", value);
#else
    setenv("MOTIFCL_ENABLE_FUSED_MLP_BACKWARD", value, 1);
#endif
}

void set_high_level_env(const char* value) {
#ifdef _WIN32
    _putenv_s("MOTIFCL_ENABLE_HIGH_LEVEL_MLP_FUSION", value);
#else
    setenv("MOTIFCL_ENABLE_HIGH_LEVEL_MLP_FUSION", value, 1);
#endif
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        constexpr int Rows = 3;
        constexpr int Channels = 4;
        constexpr int Hidden = 5;

        std::vector<float> x_host(Rows * Channels);
        std::vector<float> norm_host(Channels);
        std::vector<float> gate_host(Channels * Hidden * 2);
        std::vector<float> down_host(Hidden * Channels);

        for (int i = 0; i < Rows * Channels; ++i) {
            x_host[static_cast<std::size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.11f;
        }
        for (int i = 0; i < Channels; ++i) {
            norm_host[static_cast<std::size_t>(i)] = 0.8f + 0.07f * static_cast<float>(i);
        }
        for (int i = 0; i < Channels * Hidden * 2; ++i) {
            gate_host[static_cast<std::size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.035f;
        }
        for (int i = 0; i < Hidden * Channels; ++i) {
            down_host[static_cast<std::size_t>(i)] = static_cast<float>((i % 9) - 4) * 0.04f;
        }

        auto X = make_tensor(backend, {Rows, Channels}, x_host);
        auto N = make_tensor(backend, {Channels}, norm_host);
        auto G = make_tensor(backend, {Channels, Hidden * 2}, gate_host);
        auto D = make_tensor(backend, {Hidden, Channels}, down_host);
        X.set_requires_grad(true);
        N.set_requires_grad(true);
        G.set_requires_grad(true);
        D.set_requires_grad(true);

        set_high_level_env("1");
        auto Y = motifcl::fused_swiglu_mlp_rmsnorm_residual(X, N, G, D);
        auto L = motifcl::sum_all(Y);
        backend.profiler.clear();
        backend.profiler.set_enabled(true);
        L.backward();
        backend.finish();
        backend.profiler.set_enabled(false);

        auto Xr = make_tensor(backend, {Rows, Channels}, x_host);
        auto Nr = make_tensor(backend, {Channels}, norm_host);
        auto Gr = make_tensor(backend, {Channels, Hidden * 2}, gate_host);
        auto Dr = make_tensor(backend, {Hidden, Channels}, down_host);
        Xr.set_requires_grad(true);
        Nr.set_requires_grad(true);
        Gr.set_requires_grad(true);
        Dr.set_requires_grad(true);

        auto Normed = motifcl::rmsnorm(Xr, Nr);
        auto Packed = motifcl::matmul(Normed, Gr);
        auto HiddenTensor = motifcl::swiglu(Packed);
        auto Mlp = motifcl::matmul(HiddenTensor, Dr);
        auto Yr = motifcl::add(Xr, Mlp);
        auto Lr = motifcl::sum_all(Yr);
        Lr.backward();
        backend.finish();

        if (!close_vec(Y.to_vector<float>(), Yr.to_vector<float>(), 1e-5f, "forward")) return 1;
        if (!X.grad() || !N.grad() || !G.grad() || !D.grad()) return 2;
        if (!Xr.grad() || !Nr.grad() || !Gr.grad() || !Dr.grad()) return 3;
        if (!close_vec(X.grad()->to_vector<float>(), Xr.grad()->to_vector<float>(), 2e-4f, "grad_x")) return 4;
        if (!close_vec(N.grad()->to_vector<float>(), Nr.grad()->to_vector<float>(), 2e-4f, "grad_norm")) return 5;
        if (!close_vec(G.grad()->to_vector<float>(), Gr.grad()->to_vector<float>(), 2e-4f, "grad_gate_up")) return 6;
        if (!close_vec(D.grad()->to_vector<float>(), Dr.grad()->to_vector<float>(), 2e-4f, "grad_down")) return 7;

        std::size_t fused_count = 0;
        std::size_t plain_swiglu_backward_count = 0;
        for (const auto& row : backend.profiler.summary()) {
            if (row.name == "swiglu_down_backward_packed_f32") fused_count = row.count;
            if (row.name == "swiglu_backward_f32") plain_swiglu_backward_count = row.count;
        }
        if (fused_count != 1 || plain_swiglu_backward_count != 0) {
            std::cerr << "expected fused SwiGLU/down backward once and plain SwiGLU backward zero, got fused="
                      << fused_count << " plain=" << plain_swiglu_backward_count << "\n";
            return 8;
        }
        set_high_level_env("0");

        {
            set_fused_env("1");
            motifcl::nn::TransformerConfig cfg;
            cfg.vocab_size = 32;
            cfg.block_size = 4;
            cfg.n_embd = 16;
            cfg.n_head = 4;
            cfg.n_kv_head = 4;
            cfg.n_layer = 1;
            cfg.mlp_hidden = 24;
            cfg.dropout = 0.0f;
            cfg.use_swiglu = true;
            cfg.use_qkv_bias = false;
            motifcl::nn::ModernTransformerBlock block(backend, cfg);
            auto input = motifcl::Tensor::randn(backend, {4, cfg.n_embd}, 0.02f);
            input.set_requires_grad(true);
            auto out = block.forward(input, 1, 4);
            auto loss = motifcl::sum_all(out);
            backend.profiler.clear();
            backend.profiler.set_enabled(true);
            loss.backward();
            backend.finish();
            backend.profiler.set_enabled(false);
            bool saw_fused = false;
            for (const auto& row : backend.profiler.summary()) {
                if (row.name == "swiglu_down_backward_packed_f32" && row.count > 0) saw_fused = true;
            }
            if (!saw_fused || !input.grad()) return 9;
            set_fused_env("0");
        }

        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
