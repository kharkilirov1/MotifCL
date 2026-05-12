#include <motifcl/motifcl.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model_path;
    std::string model_dir;
    std::string gguf_path;
    std::string config_path;
    std::string tokenizer_path;
    std::string arch_name = "auto";
    std::string quant = "none";
    std::string prompt = "Hello world";
    std::vector<std::string> weights;
    bool strict = false;
    bool random_init = false;
    bool repl = false;
    bool profile = false;
    int ctx_size = 0;
    int warmup = 1;
    int iters = 3;
    motifcl::nn::GenerateOptions gen;
};

struct BenchResult {
    double prompt_eval_ms = 0.0;
    double decode_ms = 0.0;
    int generated_tokens = 0;

    double decode_tok_s() const {
        return decode_ms > 0.0 ? 1000.0 * static_cast<double>(generated_tokens) / decode_ms : 0.0;
    }
};

double elapsed_ms(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void usage() {
    std::cerr
        << "Usage: motifcl_warm_decode_bench --model PATH|--model-dir DIR [options]\n"
        << "\n"
        << "Persistent warm benchmark: model is loaded once, warmup prompts are discarded,\n"
        << "then prompt_eval_ms and decode_tok_s are measured in-process for Ollama-style comparison.\n"
        << "\nOptions:\n"
        << "  --model PATH             HF model directory or single .gguf file\n"
        << "  --model-dir DIR          HF model directory\n"
        << "  --config PATH            config.json path (default: DIR/config.json)\n"
        << "  --weights PATH           .safetensors shard; may be repeated\n"
        << "  --tokenizer PATH         tokenizer.json/vocab path or directory\n"
        << "  --arch NAME              auto|gemma|llama|mistral|qwen2|generic_decoder|...\n"
        << "  --prompt TEXT            prompt for measured runs (default: Hello world)\n"
        << "  --max-new-tokens N       default: 32\n"
        << "  --warmup N               warm in-process runs before measuring (default: 1)\n"
        << "  --iters N                measured in-process runs (default: 3)\n"
        << "  --ctx-size N             cap runtime context/cache length below model max context\n"
        << "  --paged-kv               use paged KV cache\n"
        << "  --kv-page-size N         paged KV page size (default: 256)\n"
        << "  --quant none|q8|q4       quantize dense HF Linear/lm_head weights\n"
        << "  --cpu-sampling           download logits and sample on CPU (default: GPU sampler)\n"
        << "  --seed N                 default: 1234\n"
        << "  --repl                   keep model loaded and benchmark each stdin line\n"
        << "  --profile                print OpenCL kernel profile for the final measured run\n"
        << "  --strict                 fail if expected HF/GGUF weights are missing\n"
        << "  --random-init            allow benchmark without weights for smoke testing\n";
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string extension_lower(const std::filesystem::path& path) {
    return lower_ascii(path.extension().string());
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

Options parse_args(int argc, char** argv) {
    Options opts;
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
        if (arg == "--model") opts.model_path = require_value("--model");
        else if (arg == "--model-dir") opts.model_dir = require_value("--model-dir");
        else if (arg == "--config") opts.config_path = require_value("--config");
        else if (arg == "--weights") opts.weights.push_back(require_value("--weights"));
        else if (arg == "--tokenizer") opts.tokenizer_path = require_value("--tokenizer");
        else if (arg == "--arch") opts.arch_name = require_value("--arch");
        else if (arg == "--quant") opts.quant = require_value("--quant");
        else if (arg == "--prompt") opts.prompt = require_value("--prompt");
        else if (arg == "--max-new-tokens") opts.gen.max_new_tokens = std::stoi(require_value("--max-new-tokens"));
        else if (arg == "--warmup") opts.warmup = std::stoi(require_value("--warmup"));
        else if (arg == "--iters") opts.iters = std::stoi(require_value("--iters"));
        else if (arg == "--ctx-size") opts.ctx_size = std::stoi(require_value("--ctx-size"));
        else if (arg == "--paged-kv") opts.gen.use_paged_kv_cache = true;
        else if (arg == "--kv-page-size") opts.gen.kv_page_size = std::stoi(require_value("--kv-page-size"));
        else if (arg == "--cpu-sampling") opts.gen.gpu_greedy_sampling = false;
        else if (arg == "--seed") opts.gen.seed = static_cast<std::uint32_t>(std::stoul(require_value("--seed")));
        else if (arg == "--repl") opts.repl = true;
        else if (arg == "--profile") opts.profile = true;
        else if (arg == "--strict") opts.strict = true;
        else if (arg == "--random-init") opts.random_init = true;
        else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            std::exit(2);
        }
    }
    if (opts.warmup < 0 || opts.iters <= 0 || opts.gen.max_new_tokens < 0) {
        std::cerr << "--warmup must be >=0, --iters >0, --max-new-tokens >=0\n";
        std::exit(2);
    }
    if (opts.model_path.empty() && opts.model_dir.empty() && opts.config_path.empty()) {
        std::cerr << "--model, --model-dir or --config is required\n";
        usage();
        std::exit(2);
    }
    return opts;
}

std::vector<std::int32_t> encode_prompt(const motifcl::nn::HFTokenizer& tokenizer,
                                        const std::string& prompt,
                                        const motifcl::nn::GenerateOptions& gen) {
    auto tokens = tokenizer.encode(prompt, gen.add_bos, false);
    if (tokens.empty()) tokens.push_back(gen.bos_token_id);
    return tokens;
}

std::int32_t choose_next(const motifcl::Tensor& logits,
                         const motifcl::nn::GenerateOptions& gen,
                         std::mt19937& rng) {
    if (gen.gpu_greedy_sampling) {
        return motifcl::rowwise_sample_top_p(logits, gen.temperature, gen.top_k, gen.top_p,
                                             static_cast<std::uint32_t>(rng()))
            .to_vector<std::int32_t>()[0];
    }
    const auto cpu = logits.to_vector<float>();
    std::size_t best = 0;
    for (std::size_t i = 1; i < cpu.size(); ++i) {
        if (cpu[i] > cpu[best]) best = i;
    }
    return static_cast<std::int32_t>(best);
}

BenchResult run_modern_once(motifcl::Backend& backend,
                            motifcl::nn::ModernGPTModel& model,
                            const motifcl::nn::HFTokenizer& tokenizer,
                            const std::string& prompt,
                            motifcl::nn::GenerateOptions gen,
                            std::uint32_t seed_offset) {
    auto tokens = encode_prompt(tokenizer, prompt, gen);
    MCL_CHECK(static_cast<int>(tokens.size()) <= model.config.block_size, "prompt exceeds model block_size");
    gen.seed += seed_offset;

    motifcl::Tensor logits;
    const auto prompt_start = Clock::now();
    if (gen.use_paged_kv_cache) {
        auto caches = model.create_paged_kv_cache(backend, 1, gen.kv_page_size);
        auto input = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(tokens.size())},
                                               motifcl::DType::I32, tokens.data());
        logits = model.forward_with_cache_last_logits(input, caches);
        backend.finish();
        const auto prompt_end = Clock::now();
        std::mt19937 rng(gen.seed);
        int generated = 0;
        const auto decode_start = Clock::now();
        for (int step = 0; step < gen.max_new_tokens; ++step) {
            MCL_CHECK(static_cast<int>(tokens.size()) < model.config.block_size, "decode reached model block_size");
            const auto next = choose_next(logits, gen, rng);
            tokens.push_back(next);
            ++generated;
            if (gen.eos_token_id >= 0 && next == gen.eos_token_id) break;
            if (step + 1 >= gen.max_new_tokens) break;
            auto input_next = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, &next);
            logits = model.decode_step(input_next, caches);
        }
        backend.finish();
        const auto decode_end = Clock::now();
        return {elapsed_ms(prompt_start, prompt_end), elapsed_ms(decode_start, decode_end), generated};
    }

    auto caches = model.create_kv_cache(backend, 1);
    auto input = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(tokens.size())},
                                           motifcl::DType::I32, tokens.data());
    logits = model.forward_with_cache_last_logits(input, caches);
    backend.finish();
    const auto prompt_end = Clock::now();

    std::mt19937 rng(gen.seed);
    int generated = 0;
    const auto decode_start = Clock::now();
    for (int step = 0; step < gen.max_new_tokens; ++step) {
        MCL_CHECK(static_cast<int>(tokens.size()) < model.config.block_size, "decode reached model block_size");
        const auto next = choose_next(logits, gen, rng);
        tokens.push_back(next);
        ++generated;
        if (gen.eos_token_id >= 0 && next == gen.eos_token_id) break;
        if (step + 1 >= gen.max_new_tokens) break;
        auto input_next = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, &next);
        logits = model.decode_step(input_next, caches);
    }
    backend.finish();
    const auto decode_end = Clock::now();
    return {elapsed_ms(prompt_start, prompt_end), elapsed_ms(decode_start, decode_end), generated};
}

BenchResult run_hybrid_once(motifcl::Backend& backend,
                            motifcl::nn::HybridGPTModel& model,
                            const motifcl::nn::HFTokenizer& tokenizer,
                            const std::string& prompt,
                            motifcl::nn::GenerateOptions gen,
                            std::uint32_t seed_offset) {
    auto tokens = encode_prompt(tokenizer, prompt, gen);
    MCL_CHECK(static_cast<int>(tokens.size()) <= model.config.block_size, "prompt exceeds model block_size");
    gen.seed += seed_offset;

    auto cache = model.create_runtime_cache(backend, 1, gen.use_paged_kv_cache, gen.kv_page_size);
    const auto prompt_start = Clock::now();
    auto input = motifcl::Tensor::from_cpu(backend, {1, static_cast<int64_t>(tokens.size())},
                                           motifcl::DType::I32, tokens.data());
    auto logits = model.forward_with_cache_last_logits(input, cache);
    backend.finish();
    const auto prompt_end = Clock::now();

    std::mt19937 rng(gen.seed);
    int generated = 0;
    const auto decode_start = Clock::now();
    for (int step = 0; step < gen.max_new_tokens; ++step) {
        MCL_CHECK(static_cast<int>(tokens.size()) < model.config.block_size, "decode reached model block_size");
        const auto next = choose_next(logits, gen, rng);
        tokens.push_back(next);
        ++generated;
        if (gen.eos_token_id >= 0 && next == gen.eos_token_id) break;
        if (step + 1 >= gen.max_new_tokens) break;
        auto input_next = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, &next);
        logits = model.decode_step(input_next, cache);
    }
    backend.finish();
    const auto decode_end = Clock::now();
    return {elapsed_ms(prompt_start, prompt_end), elapsed_ms(decode_start, decode_end), generated};
}

template <typename RunFn>
void run_benchmark_loop(motifcl::Backend& backend,
                        const Options& opts,
                        RunFn&& run_once,
                        double load_ms) {
    for (int i = 0; i < opts.warmup; ++i) {
        (void)run_once(opts.prompt, static_cast<std::uint32_t>(i));
    }
    std::vector<BenchResult> results;
    results.reserve(static_cast<std::size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        if (opts.profile && i + 1 == opts.iters) {
            backend.profiler.clear();
            backend.profiler.set_enabled(true);
        }
        results.push_back(run_once(opts.prompt, static_cast<std::uint32_t>(opts.warmup + i)));
        if (opts.profile && i + 1 == opts.iters) {
            backend.finish();
            backend.profiler.set_enabled(false);
        }
    }

    double prompt_total = 0.0;
    double decode_total = 0.0;
    int token_total = 0;
    for (const auto& r : results) {
        prompt_total += r.prompt_eval_ms;
        decode_total += r.decode_ms;
        token_total += r.generated_tokens;
    }
    const double prompt_avg = prompt_total / static_cast<double>(results.size());
    const double decode_tok_s = decode_total > 0.0
        ? 1000.0 * static_cast<double>(token_total) / decode_total
        : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "load_ms=" << load_ms << "\n"
              << "warmup_runs=" << opts.warmup << "\n"
              << "iters=" << opts.iters << "\n"
              << "prompt_eval_ms=" << prompt_avg << "\n"
              << "decode_ms_total=" << decode_total << "\n"
              << "decode_tokens=" << token_total << "\n"
              << "decode_tok_s=" << decode_tok_s << "\n";

    if (opts.profile) {
        std::cerr << "OpenCL kernel profile (final measured run):\n";
        int shown = 0;
        for (const auto& row : backend.profiler.summary()) {
            if (shown++ >= 24) break;
            std::cerr << "  " << std::setw(40) << std::left << row.name
                      << " count=" << std::setw(6) << row.count
                      << " total_ms=" << std::fixed << std::setprecision(3) << row.total_ms
                      << " avg_ms=" << row.avg_ms
                      << " max_ms=" << row.max_ms << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    auto opts = parse_args(argc, argv);

    if (!opts.model_path.empty()) {
        const std::filesystem::path model_arg(opts.model_path);
        if (extension_lower(model_arg) == ".gguf") {
            opts.gguf_path = model_arg.string();
            if (opts.model_dir.empty()) opts.model_dir = model_arg.parent_path().string();
        } else if (std::filesystem::is_directory(model_arg)) {
            opts.model_dir = model_arg.string();
        } else if (opts.config_path.empty()) {
            opts.config_path = model_arg.string();
        }
    }

    const auto arch = motifcl::nn::parse_hf_architecture(opts.arch_name);
    const auto load_start = Clock::now();
    auto backend = motifcl::Backend::create_opencl();
    const std::filesystem::path model_root = opts.model_dir.empty()
        ? std::filesystem::path(opts.config_path).parent_path()
        : std::filesystem::path(opts.model_dir);
    if (opts.config_path.empty()) opts.config_path = (model_root / "config.json").string();
    if (opts.tokenizer_path.empty()) opts.tokenizer_path = model_root.string();
    if (opts.weights.empty()) opts.weights = find_safetensors(model_root);

    auto cfg = !opts.gguf_path.empty()
        ? motifcl::nn::load_hf_transformer_config_gguf(opts.gguf_path, arch)
        : motifcl::nn::load_hf_transformer_config_json(opts.config_path, arch);
    if (opts.ctx_size > 0) cfg.transformer.block_size = std::min(cfg.transformer.block_size, opts.ctx_size);
    opts.gen.bos_token_id = cfg.bos_token_id;
    opts.gen.eos_token_id = cfg.eos_token_id;
    opts.gen.pad_token_id = cfg.pad_token_id;

    const auto spec = motifcl::nn::modern_model_spec_from_config(cfg);
    const bool modern_runnable = motifcl::nn::modern_model_spec_runnable_by_modern_gpt(spec);
    const bool hybrid_runnable = !modern_runnable && motifcl::nn::modern_model_spec_runnable_by_hybrid(spec);
    if (!modern_runnable && !hybrid_runnable) {
        std::cerr << "model graph is not runnable: " << cfg.architecture_name << "\n";
        for (const auto& blocker : spec.blockers) std::cerr << "blocker: " << blocker << "\n";
        return 2;
    }

    if (hybrid_runnable) {
        if (!opts.gguf_path.empty()) {
            std::cerr << "hybrid runner currently loads HF safetensors; GGUF hybrid loading is not enabled\n";
            return 2;
        }
        if (!(opts.quant == "none" || opts.quant == "None" || opts.quant == "NONE")) {
            std::cerr << "hybrid runner expects preloaded weights; --quant applies only to ModernGPTModel\n";
            return 2;
        }
        auto model = motifcl::nn::make_hf_hybrid_transformer_model(backend, cfg);
        if (!opts.weights.empty()) {
            auto report = motifcl::nn::load_hf_hybrid_transformer_weights(backend, model, opts.weights,
                                                                          cfg, opts.strict, false);
            std::cerr << "Loaded " << report.loaded_tensors << " tensors for "
                      << cfg.architecture_name << " hybrid transformer\n";
        } else if (!opts.random_init) {
            std::cerr << "no .safetensors weights found; pass --weights or --random-init\n";
            return 2;
        }
        auto tokenizer = motifcl::nn::load_hf_tokenizer(opts.tokenizer_path, cfg);
        backend.finish();
        const double load_ms = elapsed_ms(load_start, Clock::now());
        if (opts.repl) {
            std::cerr << "MotifCL warm benchmark REPL ready; one prompt per line\n";
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                auto result = run_hybrid_once(backend, model, tokenizer, line, opts.gen, 0);
                std::cout << std::fixed << std::setprecision(3)
                          << "load_ms=" << load_ms
                          << " prompt_eval_ms=" << result.prompt_eval_ms
                          << " decode_tokens=" << result.generated_tokens
                          << " decode_tok_s=" << result.decode_tok_s() << "\n" << std::flush;
            }
            return 0;
        }
        run_benchmark_loop(backend, opts,
                           [&](const std::string& prompt, std::uint32_t seed_offset) {
                               return run_hybrid_once(backend, model, tokenizer, prompt, opts.gen, seed_offset);
                           },
                           load_ms);
        return 0;
    }

    auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);
    if (!opts.gguf_path.empty()) {
        auto report = motifcl::nn::load_hf_transformer_gguf_weights(backend, model, opts.gguf_path,
                                                                    cfg, opts.strict, false);
        std::cerr << "Loaded " << report.loaded_tensors << " GGUF tensors for "
                  << cfg.architecture_name << " modern transformer\n";
    } else if (!opts.weights.empty()) {
        auto report = motifcl::nn::load_hf_transformer_weights(backend, model, opts.weights, cfg,
                                                               opts.strict, false);
        std::cerr << "Loaded " << report.loaded_tensors << " tensors for "
                  << cfg.architecture_name << " modern transformer\n";
    } else if (!opts.random_init) {
        std::cerr << "no .safetensors weights found; pass --weights or --random-init\n";
        return 2;
    }

    if (opts.quant == "q4" || opts.quant == "Q4") {
        motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q4_0);
    } else if (opts.quant == "q8" || opts.quant == "Q8") {
        motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q8_0);
    } else if (!(opts.quant == "none" || opts.quant == "None" || opts.quant == "NONE")) {
        std::cerr << "unknown --quant value: " << opts.quant << " (expected none|q8|q4)\n";
        return 2;
    }

    auto tokenizer = !opts.gguf_path.empty() && opts.tokenizer_path == model_root.string()
        ? motifcl::nn::load_hf_tokenizer_gguf(opts.gguf_path, cfg)
        : motifcl::nn::load_hf_tokenizer(opts.tokenizer_path, cfg);
    backend.finish();
    const double load_ms = elapsed_ms(load_start, Clock::now());

    if (opts.repl) {
        std::cerr << "MotifCL warm benchmark REPL ready; one prompt per line\n";
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            auto result = run_modern_once(backend, model, tokenizer, line, opts.gen, 0);
            std::cout << std::fixed << std::setprecision(3)
                      << "load_ms=" << load_ms
                      << " prompt_eval_ms=" << result.prompt_eval_ms
                      << " decode_tokens=" << result.generated_tokens
                      << " decode_tok_s=" << result.decode_tok_s() << "\n" << std::flush;
        }
        return 0;
    }

    run_benchmark_loop(backend, opts,
                       [&](const std::string& prompt, std::uint32_t seed_offset) {
                           return run_modern_once(backend, model, tokenizer, prompt, opts.gen, seed_offset);
                       },
                       load_ms);
    return 0;
}
