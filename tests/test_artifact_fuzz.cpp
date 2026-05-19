#include <motifcl/motifcl.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

void require(bool cond, const std::string& message) {
    if (!cond) throw std::runtime_error(message);
}

template <typename Fn>
void require_motifcl_error(Fn&& fn, const std::string& message) {
    try {
        fn();
    } catch (const motifcl::Error&) {
        return;
    }
    throw std::runtime_error(message);
}

template <typename T>
void write_native(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open fuzz artifact");
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    require(out.good(), "failed to write fuzz artifact");
}

void write_payload_header(std::ostream& out,
                          motifcl::DType dtype,
                          std::vector<std::int64_t> dims,
                          std::uint64_t nbytes,
                          float quant_scale,
                          std::int32_t has_quant_scales,
                          std::int32_t quant_scale_axis,
                          std::int64_t quant_block_size) {
    write_native(out, static_cast<std::int32_t>(dtype));
    write_native(out, static_cast<std::uint64_t>(dims.size()));
    for (auto dim : dims) write_native(out, dim);
    write_native(out, nbytes);
    write_native(out, quant_scale);
    write_native(out, has_quant_scales);
    write_native(out, quant_scale_axis);
    write_native(out, quant_block_size);
}

void write_bad_axis1_q8_mclt(const std::filesystem::path& path, std::int64_t rows, std::int64_t cols) {
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open malformed mclt");
    const char magic[8] = {'M', 'C', 'L', 'T', 'E', 'N', '2', '\0'};
    out.write(magic, sizeof(magic));
    write_payload_header(out, motifcl::DType::Q8_0, {rows, cols}, static_cast<std::uint64_t>(rows * cols),
                         1.0f, 1, 1, 0);
    write_payload_header(out, motifcl::DType::F32, {1}, 4, 1.0f, 0, -1, 0);
    const float single_scale = 1.0f;
    out.write(reinterpret_cast<const char*>(&single_scale), sizeof(single_scale));
    std::vector<std::uint8_t> qbytes(static_cast<std::size_t>(rows * cols), 0);
    out.write(reinterpret_cast<const char*>(qbytes.data()), static_cast<std::streamsize>(qbytes.size()));
    require(out.good(), "failed to write malformed mclt");
}

void fuzz_mclt(motifcl::Backend& backend, const std::filesystem::path& root) {
    for (std::size_t size : {0u, 1u, 7u, 8u, 16u, 31u, 63u}) {
        std::vector<std::uint8_t> bytes(size);
        for (std::size_t i = 0; i < size; ++i) bytes[i] = static_cast<std::uint8_t>((i * 37u + size) & 0xffu);
        const auto path = root / ("fuzz_truncated_" + std::to_string(size) + ".mclt");
        write_bytes(path, bytes);
        require_motifcl_error([&] { (void)motifcl::load_tensor(backend, path.string()); },
                              "malformed .mclt was accepted");
    }

    const auto short_scales = root / "fuzz_short_axis1_scales.mclt";
    write_bad_axis1_q8_mclt(short_scales, 2, 4);
    require_motifcl_error([&] { (void)motifcl::load_tensor(backend, short_scales.string()); },
                          ".mclt with short axis-1 quant scales was accepted");
}

void fuzz_gguf(const std::filesystem::path& root) {
    const auto wrong_magic = root / "fuzz_wrong_magic.gguf";
    write_bytes(wrong_magic, {0, 1, 2, 3, 4, 5, 6, 7});
    require_motifcl_error([&] { (void)motifcl::gguf::File::open(wrong_magic.string()); },
                          "GGUF wrong magic was accepted");

    const auto truncated_meta = root / "fuzz_truncated_metadata.gguf";
    {
        std::ofstream out(truncated_meta, std::ios::binary);
        write_native(out, static_cast<std::uint32_t>(0x46554747u)); // GGUF
        write_native(out, static_cast<std::uint32_t>(3u));
        write_native(out, static_cast<std::uint64_t>(0u)); // tensors
        write_native(out, static_cast<std::uint64_t>(1u)); // metadata entries, but payload omitted
    }
    require_motifcl_error([&] { (void)motifcl::gguf::File::open(truncated_meta.string()); },
                          "truncated GGUF metadata was accepted");
}

void fuzz_quantized_checkpoint(motifcl::Backend& backend, const std::filesystem::path& root) {
    motifcl::nn::GemmaConfig cfg;
    cfg.vocab_size = 8;
    cfg.max_position_embeddings = 8;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 8;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 2;
    cfg.eos_token_id = 7;
    auto model = motifcl::nn::make_gemma_model(backend, cfg);
    model.enable_quantized_inference(motifcl::DType::Q4_0);
    const auto dir = root / "fuzz_quantized_checkpoint";
    motifcl::save_quantized_transformer_checkpoint(model, dir.string(), motifcl::DType::Q4_0);
    write_bad_axis1_q8_mclt(dir / "lm_head.qweight.mclt", cfg.hidden_size, cfg.vocab_size);
    auto rejected = motifcl::nn::make_gemma_model(backend, cfg);
    require_motifcl_error([&] {
        motifcl::load_quantized_transformer_checkpoint(rejected, backend, dir.string());
    }, "malformed quantized checkpoint directory was accepted");
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        const auto root = std::filesystem::current_path() / "artifact_fuzz_tmp";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        try {
            fuzz_mclt(backend, root);
            fuzz_gguf(root);
            fuzz_quantized_checkpoint(backend, root);
        } catch (...) {
            std::filesystem::remove_all(root);
            throw;
        }
        std::filesystem::remove_all(root);
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
