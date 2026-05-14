#include <motifcl/motifcl.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "Usage: motifcl_generate_transformer --model PATH|--model-dir DIR [--prompt TEXT|--repl] [options]\n"
        << "\nOptions:\n"
        << "  --model PATH             HF-style model directory or single .gguf file\n"
        << "  --config PATH            HF-style config.json path (default: DIR/config.json)\n"
        << "  --weights PATH           .safetensors shard; may be repeated (default: DIR/*.safetensors)\n"
        << "  --tokenizer PATH         tokenizer.json/vocab path or directory (default: DIR)\n"
        << "  --arch NAME              auto|gemma|llama|mistral|qwen2|generic_decoder|...\n"
        << "  --inspect                print compatibility/readiness probe and exit\n"
        << "  --list-architectures     print runner architecture registry and exit\n"
        << "  --system TEXT            system chat message; activates chat templating\n"
        << "  --user TEXT              user chat message; activates chat templating\n"
        << "  --message ROLE:TEXT      append chat message; may be repeated\n"
        << "  --chat-template KIND     auto|none|generic|chatml|llama2|llama3|mistral|gemma\n"
        << "  --no-generation-prompt   do not append assistant generation marker in chat mode\n"
        << "  --max-new-tokens N       default: 32\n"
        << "  --temperature FLOAT      default: 0.0 greedy\n"
        << "  --top-k N                default: 0\n"
        << "  --top-p FLOAT            default: 1.0\n"
        << "  --quant none|q8|q4       quantize Linear/lm_head weights for no-grad inference\n"
        << "  --ctx-size N             cap runtime context/cache length below model max context\n"
        << "  --no-prefill             force prompt token-by-token instead of one cached prefill\n"
        << "  --force-prefill          force one cached [T] prefill and disable adaptive prefill\n"
        << "  --disable-adaptive-prefill\n"
        << "                            disable default small GGUF K-quant streaming prefill\n"
        << "  --adaptive-prefill-max-tokens N\n"
        << "                            default: 24; env override: MOTIFCL_ADAPTIVE_STREAMING_PREFILL_MAX_TOKENS\n"
        << "  --paged-kv               use paged KV cache for single-prompt generation\n"
        << "  --kv-page-size N         paged KV page size (default: 256)\n"
        << "  --repl                   keep model loaded and read one prompt per stdin line\n"
        << "  --jsonl-repl             persistent machine REPL: read JSON/plain lines, supports stream=true deltas\n"
        << "  --completion-only        print only newly generated text, not prompt+completion\n"
        << "  --cpu-sampling           download logits and sample on CPU instead of GPU sampler\n"
        << "  --profile                print top OpenCL kernel timings for generation\n"
        << "  --seed N                 default: 1234\n"
        << "  --strict                 fail if expected HF weights are missing\n"
        << "  --random-init            allow generation without weights for smoke testing\n";
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

void print_architecture_registry() {
    std::cout << "MotifCL modern model runner architecture registry\n";
    for (const auto& info : motifcl::nn::hf_architecture_registry()) {
        std::cout << "- " << info.name
                  << " hf=" << (info.supports_hf_safetensors ? "yes" : "no")
                  << " gguf=" << (info.supports_gguf ? "yes" : "no")
                  << " chat=" << (info.supports_chat_template ? "yes" : "no");
        if (!info.aliases.empty()) {
            std::cout << " aliases=";
            for (std::size_t i = 0; i < info.aliases.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << info.aliases[i];
            }
        }
        std::cout << "\n";
        if (!info.notes.empty()) std::cout << "  notes: " << info.notes << "\n";
        for (const auto& blocker : info.blockers) std::cout << "  blocker: " << blocker << "\n";
    }
}

motifcl::nn::HFChatMessage parse_chat_message(const std::string& value) {
    const auto colon = value.find(':');
    if (colon == std::string::npos) return {"user", value};
    return {value.substr(0, colon), value.substr(colon + 1)};
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string json_unescape(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\' || i + 1 >= value.size()) {
            out.push_back(value[i]);
            continue;
        }
        const char c = value[++i];
        switch (c) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

bool json_string_field(const std::string& text, const std::string& key, std::string& out) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, re)) return false;
    out = json_unescape(match[1].str());
    return true;
}

bool json_number_field(const std::string& text, const std::string& key, double& out) {
    const std::regex re("\"" + key + R"("\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?))");
    std::smatch match;
    if (!std::regex_search(text, match, re)) return false;
    out = std::stod(match[1].str());
    return true;
}

bool json_bool_field(const std::string& text, const std::string& key, bool& out) {
    const std::regex re("\"" + key + R"("\s*:\s*(true|false))", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(text, match, re)) return false;
    out = lower_ascii(match[1].str()) == "true";
    return true;
}

bool looks_like_json_object(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    return first != std::string::npos && text[first] == '{';
}

motifcl::nn::GenerateOptions options_from_json_line(const std::string& line,
                                                    motifcl::nn::GenerateOptions base) {
    double number = 0.0;
    bool flag = false;
    if (json_number_field(line, "max_new_tokens", number) ||
        json_number_field(line, "num_predict", number)) {
        base.max_new_tokens = static_cast<int>(number);
    }
    if (json_number_field(line, "temperature", number)) base.temperature = static_cast<float>(number);
    if (json_number_field(line, "top_k", number)) base.top_k = static_cast<int>(number);
    if (json_number_field(line, "top_p", number)) base.top_p = static_cast<float>(number);
    if (json_number_field(line, "seed", number)) base.seed = static_cast<std::uint32_t>(number);
    if (json_bool_field(line, "ignore_eos", flag) && flag) base.eos_token_id = -1;
    if (json_bool_field(line, "cpu_sampling", flag)) base.gpu_greedy_sampling = !flag;
    return base;
}

std::string prompt_from_json_or_plain(const std::string& line) {
    if (!looks_like_json_object(line)) return line;
    std::string value;
    if (json_string_field(line, "prompt", value)) return value;
    if (json_string_field(line, "input", value)) return value;
    return {};
}

std::string id_from_json_or_empty(const std::string& line) {
    if (!looks_like_json_object(line)) return {};
    std::string value;
    json_string_field(line, "id", value);
    return value;
}

bool stream_from_json_or_false(const std::string& line) {
    if (!looks_like_json_object(line)) return false;
    bool value = false;
    return json_bool_field(line, "stream", value) && value;
}

void write_json_delta(const std::string& request_id, std::int32_t token_id, const std::string& delta) {
    std::cout << "{\"ok\":true";
    if (!request_id.empty()) std::cout << ",\"id\":\"" << json_escape(request_id) << "\"";
    std::cout << ",\"token\":" << token_id
              << ",\"delta\":\"" << json_escape(delta) << "\""
              << ",\"done\":false}\n" << std::flush;
}

void write_json_final(const std::string& request_id,
                      const std::string& response,
                      double total_ms,
                      bool include_done) {
    std::cout << "{\"ok\":true";
    if (!request_id.empty()) std::cout << ",\"id\":\"" << json_escape(request_id) << "\"";
    std::cout << ",\"response\":\"" << json_escape(response) << "\"";
    if (include_done) std::cout << ",\"done\":true";
    std::cout << ",\"total_ms\":" << std::fixed << std::setprecision(3) << total_ms
              << "}\n" << std::flush;
}

void write_json_error(const std::string& request_id, const std::string& error, bool include_done) {
    std::cout << "{\"ok\":false";
    if (!request_id.empty()) std::cout << ",\"id\":\"" << json_escape(request_id) << "\"";
    std::cout << ",\"error\":\"" << json_escape(error) << "\"";
    if (include_done) std::cout << ",\"done\":true";
    std::cout << "}\n" << std::flush;
}

std::string generate_modern_output(motifcl::Backend& backend,
                                   motifcl::nn::ModernGPTModel& model,
                                   const motifcl::nn::HFTokenizer& tokenizer,
                                   const std::string& prompt,
                                   motifcl::nn::GenerateOptions options,
                                   bool completion_only,
                                   const motifcl::nn::TextTokenCallback& token_callback = {}) {
    if (!completion_only) {
        return motifcl::nn::generate_hf_text(backend, model, tokenizer, prompt, options, token_callback);
    }
    auto prompt_ids = tokenizer.encode(prompt, options.add_bos, false);
    auto local_options = options;
    local_options.add_bos = false;
    if (local_options.eos_token_id < 0) local_options.eos_token_id = tokenizer.eos_token_id();
    std::vector<std::int32_t> generated_ids;
    std::string emitted_text;
    motifcl::nn::TokenCallback id_callback;
    if (token_callback) {
        id_callback = [&](std::int32_t token_id) {
            generated_ids.push_back(token_id);
            const auto current = tokenizer.decode(generated_ids, true);
            std::string delta;
            if (current.rfind(emitted_text, 0) == 0) {
                delta = current.substr(emitted_text.size());
            } else {
                delta = tokenizer.decode({token_id}, true);
            }
            emitted_text = current;
            token_callback(token_id, delta);
        };
    }
    auto all_ids = motifcl::nn::generate(backend, model, prompt_ids, local_options, id_callback);
    const auto begin = std::min(prompt_ids.size(), all_ids.size());
    std::vector<std::int32_t> generated(all_ids.begin() + static_cast<std::ptrdiff_t>(begin), all_ids.end());
    return tokenizer.decode(generated, true);
}

std::string completion_by_prefix_strip(const std::string& prompt, const std::string& text) {
    return text.rfind(prompt, 0) == 0 ? text.substr(prompt.size()) : text;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string model_dir;
    std::string gguf_path;
    std::string config_path;
    std::string tokenizer_path;
    std::string prompt;
    std::string arch_name = "auto";
    std::string quant = "none";
    std::string system_message;
    std::string user_message;
    std::vector<std::string> weights;
    std::vector<motifcl::nn::HFChatMessage> chat_messages;
    motifcl::nn::HFChatTemplateKind chat_template = motifcl::nn::HFChatTemplateKind::Auto;
    bool strict = false;
    bool random_init = false;
    bool inspect = false;
    bool list_architectures = false;
    bool add_generation_prompt = true;
    bool profile = false;
    bool repl = false;
    bool jsonl_repl = false;
    bool completion_only = false;
    int ctx_size = 0;
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
        if (arg == "--model") model_path = require_value("--model");
        else if (arg == "--model-dir") model_dir = require_value("--model-dir");
        else if (arg == "--config") config_path = require_value("--config");
        else if (arg == "--weights") weights.push_back(require_value("--weights"));
        else if (arg == "--tokenizer") tokenizer_path = require_value("--tokenizer");
        else if (arg == "--prompt") prompt = require_value("--prompt");
        else if (arg == "--arch") arch_name = require_value("--arch");
        else if (arg == "--inspect") inspect = true;
        else if (arg == "--list-architectures") list_architectures = true;
        else if (arg == "--system") system_message = require_value("--system");
        else if (arg == "--user") user_message = require_value("--user");
        else if (arg == "--message") chat_messages.push_back(parse_chat_message(require_value("--message")));
        else if (arg == "--chat-template") chat_template = motifcl::nn::parse_hf_chat_template_kind(require_value("--chat-template"));
        else if (arg == "--no-generation-prompt") add_generation_prompt = false;
        else if (arg == "--max-new-tokens") options.max_new_tokens = std::stoi(require_value("--max-new-tokens"));
        else if (arg == "--temperature") options.temperature = std::stof(require_value("--temperature"));
        else if (arg == "--top-k") options.top_k = std::stoi(require_value("--top-k"));
        else if (arg == "--top-p") options.top_p = std::stof(require_value("--top-p"));
        else if (arg == "--quant") quant = require_value("--quant");
        else if (arg == "--ctx-size") ctx_size = std::stoi(require_value("--ctx-size"));
        else if (arg == "--no-prefill") options.prefill_prompt = false;
        else if (arg == "--force-prefill") {
            options.prefill_prompt = true;
            options.adaptive_prefill = false;
        }
        else if (arg == "--disable-adaptive-prefill") options.adaptive_prefill = false;
        else if (arg == "--adaptive-prefill-max-tokens") {
            options.adaptive_prefill_max_tokens = std::stoi(require_value("--adaptive-prefill-max-tokens"));
        }
        else if (arg == "--paged-kv") options.use_paged_kv_cache = true;
        else if (arg == "--kv-page-size") options.kv_page_size = std::stoi(require_value("--kv-page-size"));
        else if (arg == "--repl") repl = true;
        else if (arg == "--jsonl-repl") {
            jsonl_repl = true;
            repl = true;
        }
        else if (arg == "--completion-only") completion_only = true;
        else if (arg == "--cpu-sampling") options.gpu_greedy_sampling = false;
        else if (arg == "--profile") profile = true;
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

    if (options.max_new_tokens < 0 || options.adaptive_prefill_max_tokens < 0) {
        std::cerr << "--max-new-tokens and --adaptive-prefill-max-tokens must be >=0\n";
        return 2;
    }

    if (list_architectures) {
        print_architecture_registry();
        return 0;
    }

    if (!model_path.empty()) {
        const std::filesystem::path model_arg(model_path);
        if (extension_lower(model_arg) == ".gguf") {
            gguf_path = model_arg.string();
            if (model_dir.empty()) model_dir = model_arg.parent_path().string();
        } else if (std::filesystem::is_directory(model_arg)) {
            model_dir = model_arg.string();
        } else if (config_path.empty()) {
            config_path = model_arg.string();
        }
    }

    if (model_dir.empty() && config_path.empty() && gguf_path.empty()) {
        std::cerr << "--model, --model-dir or --config is required\n";
        usage();
        return 2;
    }
    const auto arch = motifcl::nn::parse_hf_architecture(arch_name);
    const std::string probe_source = !gguf_path.empty()
        ? gguf_path
        : (!model_path.empty() ? model_path : (!model_dir.empty() ? model_dir : config_path));
    if (inspect) {
        const auto probe = motifcl::nn::probe_hf_transformer_model(probe_source, arch);
        std::cout << motifcl::nn::format_hf_model_probe(probe);
        if (probe.has_config) {
            std::cout << motifcl::nn::format_modern_model_spec(
                motifcl::nn::modern_model_spec_from_config(probe.config));
        }
        return 0;
    }

    const bool chat_requested = !system_message.empty() || !user_message.empty() || !chat_messages.empty() ||
                                chat_template != motifcl::nn::HFChatTemplateKind::Auto;
    if (prompt.empty() && !chat_requested && !repl) {
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
    auto cfg = !gguf_path.empty()
        ? motifcl::nn::load_hf_transformer_config_gguf(gguf_path, arch)
        : motifcl::nn::load_hf_transformer_config_json(config_path, arch);
    if (ctx_size > 0) cfg.transformer.block_size = std::min(cfg.transformer.block_size, ctx_size);
    const auto format = !gguf_path.empty()
        ? motifcl::nn::HFModelFormat::GGUF
        : motifcl::nn::HFModelFormat::HuggingFaceDirectory;
    const auto spec = motifcl::nn::modern_model_spec_from_config(cfg);
    const bool modern_runnable = motifcl::nn::modern_model_spec_runnable_by_modern_gpt(spec);
    const bool hybrid_runnable = !modern_runnable && motifcl::nn::modern_model_spec_runnable_by_hybrid(spec);
    if (!hybrid_runnable && !motifcl::nn::hf_architecture_supported_now(cfg.architecture, format)) {
        std::cerr << "architecture is not runnable by the current ModernGPTModel runner: "
                  << cfg.architecture_name << "\n";
        for (const auto& blocker : motifcl::nn::hf_architecture_blockers(cfg.architecture)) {
            std::cerr << "blocker: " << blocker << "\n";
        }
        return 2;
    }
    if (!modern_runnable && !hybrid_runnable) {
        std::cerr << "model graph is not runnable by the current ModernGPTModel runner: "
                  << cfg.architecture_name << "\n";
        for (const auto& blocker : spec.blockers) std::cerr << "blocker: " << blocker << "\n";
        return 2;
    }

    const auto resolved_template = chat_template == motifcl::nn::HFChatTemplateKind::Auto
        ? motifcl::nn::infer_hf_chat_template_kind(cfg.architecture, model_root.string())
        : chat_template;
    auto render_prompt = [&](const std::string& runtime_prompt, bool append_runtime_prompt) {
        if (!chat_requested) return append_runtime_prompt ? runtime_prompt : prompt;
        std::vector<motifcl::nn::HFChatMessage> messages;
        if (!system_message.empty()) messages.push_back({"system", system_message});
        messages.insert(messages.end(), chat_messages.begin(), chat_messages.end());
        if (!user_message.empty()) messages.push_back({"user", user_message});
        if (!prompt.empty()) messages.push_back({"user", prompt});
        if (append_runtime_prompt && !runtime_prompt.empty()) messages.push_back({"user", runtime_prompt});
        return motifcl::nn::apply_hf_chat_template(messages, cfg.architecture, resolved_template,
                                                   add_generation_prompt);
    };
    auto print_profile_summary = [&]() {
        backend.finish();
        backend.profiler.set_enabled(false);
        std::cerr << "OpenCL kernel profile (generation only):\n";
        int shown = 0;
        for (const auto& row : backend.profiler.summary()) {
            if (shown++ >= 20) break;
            std::cerr << "  " << std::setw(36) << std::left << row.name
                      << " count=" << std::setw(6) << row.count
                      << " total_ms=" << std::fixed << std::setprecision(3) << row.total_ms
                      << " avg_ms=" << row.avg_ms
                      << " max_ms=" << row.max_ms << "\n";
        }
    };

    if (hybrid_runnable) {
        if (!gguf_path.empty()) {
            std::cerr << "hybrid Qwen/MoE runner currently loads HF safetensors; GGUF hybrid weight loading is not enabled\n";
            return 2;
        }
        if (!(quant == "none" || quant == "None" || quant == "NONE")) {
            std::cerr << "hybrid runner expects preloaded F32/BF16/F16 safetensors; --quant is only wired for ModernGPTModel\n";
            return 2;
        }
        auto hybrid_model = motifcl::nn::make_hf_hybrid_transformer_model(backend, cfg);
        if (!weights.empty()) {
            auto report = motifcl::nn::load_hf_hybrid_transformer_weights(backend, hybrid_model, weights, cfg, strict, false);
            std::cerr << "Loaded " << report.loaded_tensors << " tensors for "
                      << cfg.architecture_name << " hybrid transformer";
            if (!report.missing.empty()) std::cerr << " (missing: " << report.missing.size() << ")";
            if (!report.unexpected.empty()) std::cerr << " (unexpected: " << report.unexpected.size() << ")";
            std::cerr << "\n";
        } else if (!random_init) {
            std::cerr << "no .safetensors weights found; pass --weights or --random-init\n";
            return 2;
        } else {
            std::cerr << "warning: running random initialized hybrid transformer\n";
        }

        auto tokenizer = motifcl::nn::load_hf_tokenizer(tokenizer_path, cfg);
        options.bos_token_id = cfg.bos_token_id;
        options.eos_token_id = cfg.eos_token_id;
        options.pad_token_id = cfg.pad_token_id;
        if (chat_requested) {
            std::cerr << "Applied chat template: " << motifcl::nn::hf_chat_template_name(resolved_template) << "\n";
        }
        if (repl) {
            std::cerr << "MotifCL persistent " << (jsonl_repl ? "JSONL " : "")
                      << "REPL ready; one prompt per line, Ctrl+Z/EOF to exit\n";
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                const std::string request_id = id_from_json_or_empty(line);
                const std::string runtime_prompt = prompt_from_json_or_plain(line);
                auto request_options = looks_like_json_object(line)
                    ? options_from_json_line(line, options)
                    : options;
                const bool stream_response = jsonl_repl && stream_from_json_or_false(line);
                try {
                    if (profile) {
                        backend.profiler.clear();
                        backend.profiler.set_enabled(true);
                    }
                    const auto start = std::chrono::steady_clock::now();
                    const auto rendered = render_prompt(runtime_prompt, true);
                    motifcl::nn::TextTokenCallback token_callback;
                    if (stream_response) {
                        token_callback = [&](std::int32_t token_id, const std::string& delta) {
                            write_json_delta(request_id, token_id, delta);
                        };
                    }
                    auto text = motifcl::nn::generate_hf_hybrid_text(backend, hybrid_model, tokenizer,
                                                                      rendered, request_options, token_callback);
                    if (completion_only) text = completion_by_prefix_strip(rendered, text);
                    backend.finish();
                    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start).count();
                    if (jsonl_repl) {
                        write_json_final(request_id, text, static_cast<double>(elapsed_us) / 1000.0, stream_response);
                    } else {
                        std::cout << text << "\n" << std::flush;
                    }
                    if (profile) print_profile_summary();
                } catch (const std::exception& e) {
                    if (jsonl_repl) {
                        write_json_error(request_id, e.what(), stream_response);
                    } else {
                        std::cerr << "generation failed: " << e.what() << "\n";
                    }
                }
            }
            return 0;
        }
        const std::string final_prompt = render_prompt(prompt, false);
        if (profile) {
            backend.profiler.clear();
            backend.profiler.set_enabled(true);
        }
        auto text = motifcl::nn::generate_hf_hybrid_text(backend, hybrid_model, tokenizer, final_prompt, options);
        if (completion_only) text = completion_by_prefix_strip(final_prompt, text);
        std::cout << text << "\n";
        if (profile) print_profile_summary();
        return 0;
    }
    auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);

    if (!gguf_path.empty()) {
        auto report = motifcl::nn::load_hf_transformer_gguf_weights(backend, model, gguf_path, cfg, strict, false);
        std::cerr << "Loaded " << report.loaded_tensors << " GGUF tensors for "
                  << cfg.architecture_name << " modern transformer";
        if (!report.missing.empty()) std::cerr << " (missing: " << report.missing.size() << ")";
        if (!report.unexpected.empty()) std::cerr << " (unexpected: " << report.unexpected.size() << ")";
        std::cerr << "\n";
    } else if (!weights.empty()) {
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

    auto tokenizer = !gguf_path.empty() && tokenizer_path == model_root.string()
        ? motifcl::nn::load_hf_tokenizer_gguf(gguf_path, cfg)
        : motifcl::nn::load_hf_tokenizer(tokenizer_path, cfg);
    options.bos_token_id = cfg.bos_token_id;
    options.eos_token_id = cfg.eos_token_id;
    options.pad_token_id = cfg.pad_token_id;
    if (chat_requested) {
        std::cerr << "Applied chat template: " << motifcl::nn::hf_chat_template_name(resolved_template) << "\n";
    }
    if (repl) {
        std::cerr << "MotifCL persistent " << (jsonl_repl ? "JSONL " : "")
                  << "REPL ready; one prompt per line, Ctrl+Z/EOF to exit\n";
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            const std::string request_id = id_from_json_or_empty(line);
            const std::string runtime_prompt = prompt_from_json_or_plain(line);
            auto request_options = looks_like_json_object(line)
                ? options_from_json_line(line, options)
                : options;
            const bool stream_response = jsonl_repl && stream_from_json_or_false(line);
            try {
                if (profile) {
                    backend.profiler.clear();
                    backend.profiler.set_enabled(true);
                }
                const auto start = std::chrono::steady_clock::now();
                motifcl::nn::TextTokenCallback token_callback;
                if (stream_response) {
                    token_callback = [&](std::int32_t token_id, const std::string& delta) {
                        write_json_delta(request_id, token_id, delta);
                    };
                }
                const auto text = generate_modern_output(backend, model, tokenizer,
                                                         render_prompt(runtime_prompt, true),
                                                         request_options, completion_only, token_callback);
                backend.finish();
                const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (jsonl_repl) {
                    write_json_final(request_id, text, static_cast<double>(elapsed_us) / 1000.0, stream_response);
                } else {
                    std::cout << text << "\n" << std::flush;
                }
                if (profile) print_profile_summary();
            } catch (const std::exception& e) {
                if (jsonl_repl) {
                    write_json_error(request_id, e.what(), stream_response);
                } else {
                    std::cerr << "generation failed: " << e.what() << "\n";
                }
            }
        }
        return 0;
    }
    const std::string final_prompt = render_prompt(prompt, false);
    if (profile) {
        backend.profiler.clear();
        backend.profiler.set_enabled(true);
    }
    std::cout << generate_modern_output(backend, model, tokenizer, final_prompt, options, completion_only) << "\n";
    if (profile) print_profile_summary();
    return 0;
}
