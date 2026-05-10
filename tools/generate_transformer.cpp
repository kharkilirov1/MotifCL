#include <motifcl/motifcl.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "Usage: motifcl_generate_transformer --model-dir DIR --prompt TEXT [options]\n"
        << "\nOptions:\n"
        << "  --config PATH            HF-style config.json path (default: DIR/config.json)\n"
        << "  --weights PATH           .safetensors shard; may be repeated (default: DIR/*.safetensors)\n"
        << "  --tokenizer PATH         tokenizer.json/vocab path or directory (default: DIR)\n"
        << "  --arch NAME              auto|gemma|llama|mistral|qwen2|generic_decoder\n"
        << "  --max-new-tokens N       default: 32\n"
        << "  --temperature FLOAT      default: 0.0 greedy\n"
        << "  --top-k N                default: 0\n"
        << "  --quant none|q8|q4       quantize Linear/lm_head weights for no-grad inference\n"
        << "  --no-prefill             decode prompt token-by-token instead of one cached prefill\n"
        << "  --cpu-sampling           download logits and sample on CPU instead of GPU sampler\n"
        << "  --seed N                 default: 1234\n"
        << "  --strict                 fail if expected HF weights are missing\n"
        << "  --random-init            allow generation without weights for smoke testing\n";
}

std::vector<std::string> find_safetensors(const std::filesystem::path& dir) {
    std::vector<std::string> out;
    if (!std::filesystem::is_directory(dir)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            out.push_back(entry.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string config_path;
    std::string tokenizer_path;
    std::string prompt;
    std::string arch_name = "auto";
    std::string quant = "none";
    std::vector<std::string> weights;
    bool strict = false;
    bool random_init = false;
    motifcl::nn::GenerateOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--model-dir") model_dir = require_value("--model-dir");
        else if (arg == "--config") config_path = require_value("--config");
        else if (arg == "--weights") weights.push_back(require_value("--weights"));
        else if (arg == "--tokenizer") tokenizer_path = require_value("--tokenizer");
        else if (arg == "--prompt") prompt = require_value("--prompt");
        else if (arg == "--arch") arch_name = require_value("--arch");
        else if (arg == "--max-new-tokens") options.max_new_tokens = std::stoi(require_value("--max-new-tokens"));
        else if (arg == "--temperature") options.temperature = std::stof(require_value("--temperature"));
        else if (arg == "--top-k") options.top_k = std::stoi(require_value("--top-k"));
        else if (arg == "--top-p") options.top_p = std::stof(require_value("--top-p"));
        else if (arg == "--quant") quant = require_value("--quant");
        else if (arg == "--no-prefill") options.prefill_prompt = false;
        else if (arg == "--cpu-sampling") options.gpu_greedy_sampling = false;
        else if (arg == "--seed") options.seed = static_cast<std::uint32_t>(std::stoul(require_value("--seed")));
        else if (arg == "--strict") strict = true;
        else if (arg == "--random-init") random_init = true;
        else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            return 2;
        }
    }

    if (model_dir.empty() && config_path.empty()) {
        std::cerr << "--model-dir or --config is required\n";
        usage();
        return 2;
    }
    if (prompt.empty()) {
        std::cerr << "--prompt is required\n";
        usage();
        return 2;
    }

    const std::filesystem::path model_root = model_dir.empty()
        ? std::filesystem::path(config_path).parent_path()
        : std::filesystem::path(model_dir);
    if (config_path.empty()) config_path = (model_root / "config.json").string();
    if (tokenizer_path.empty()) tokenizer_path = model_root.string();
    if (weights.empty()) weights = find_safetensors(model_root);

    auto backend = motifcl::Backend::create_opencl();
    const auto arch = motifcl::nn::parse_hf_architecture(arch_name);
    auto cfg = motifcl::nn::load_hf_transformer_config_json(config_path, arch);
    auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);

    if (!weights.empty()) {
        auto report = motifcl::nn::load_hf_transformer_weights(backend, model, weights, cfg, strict, false);
        std::cerr << "Loaded " << report.loaded_tensors << " tensors for "
                  << cfg.architecture_name << " modern transformer";
        if (!report.missing.empty()) std::cerr << " (missing: " << report.missing.size() << ")";
        if (!report.unexpected.empty()) std::cerr << " (unexpected: " << report.unexpected.size() << ")";
        std::cerr << "\n";
    } else if (!random_init) {
        std::cerr << "no .safetensors weights found; pass --weights or --random-init\n";
        return 2;
    } else {
        std::cerr << "warning: running random initialized modern transformer\n";
    }

    if (quant == "q4" || quant == "Q4") {
        motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q4_0);
        std::cerr << "Enabled Q4_0 quantized modern transformer inference\n";
    } else if (quant == "q8" || quant == "Q8") {
        motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q8_0);
        std::cerr << "Enabled Q8_0 quantized modern transformer inference\n";
    } else if (!(quant == "none" || quant == "None" || quant == "NONE")) {
        std::cerr << "unknown --quant value: " << quant << " (expected none|q8|q4)\n";
        return 2;
    }

    auto tokenizer = motifcl::nn::load_hf_tokenizer(tokenizer_path, cfg);
    options.bos_token_id = cfg.bos_token_id;
    options.eos_token_id = cfg.eos_token_id;
    options.pad_token_id = cfg.pad_token_id;
    std::cout << motifcl::nn::generate_hf_text(backend, model, tokenizer, prompt, options) << "\n";
    return 0;
}
