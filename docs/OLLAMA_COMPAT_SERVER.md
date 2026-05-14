# Ollama-compatible hot server

`tools/motifcl_ollama_server.py` is a small stdlib-only HTTP front-end for MotifCL inference.
It closes the main product-UX gap where every CLI invocation previously paid model load, tokenizer setup,
OpenCL kernel build, and one-time warmup cost. The server keeps one
`motifcl_generate_transformer --jsonl-repl --completion-only` worker alive, waits until the worker reports
REPL readiness, runs a configurable one-token warmup by default, and only then starts accepting HTTP traffic.

This is not a replacement for Ollama's model registry or lifecycle manager yet; it is a compatible hot local
serving path for benchmarking and app integration.

## Build the runner

```bash
cmake -S . -B build -DMOTIFCL_BUILD_TOOLS=ON
cmake --build build --target motifcl_generate_transformer -j
```

The server auto-detects common build-tree runner locations such as
`build_kquant_prefill/tools/motifcl_generate_transformer.exe` and `build/tools/motifcl_generate_transformer.exe`.
Use `--runner PATH` to override it.

## Start a hot server

```bash
python tools/motifcl_ollama_server.py \
  --model ./model.gguf \
  --name motifcl-local:latest \
  --port 11435 \
  --ctx-size 512 \
  --max-new-tokens 128
```

Useful flags:

- `--host`, `--port` — bind address.
- `--name` — public API model name; defaults to the model path stem.
- `--runner` — explicit `motifcl_generate_transformer` binary.
- `--ctx-size`, `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p` — default generation options.
- `--arch`, `--tokenizer`, `--random-init` — forwarded to the runner.
- `--no-prefill`, `--force-prefill`, `--disable-adaptive-prefill`, `--cpu-sampling` — forwarded inference toggles.
- `--extra-arg ARG` — pass a raw extra runner argument; repeatable.
- `--startup-timeout SECONDS` — maximum wait for `REPL ready` before warning/failing.
- `--warmup-prompt TEXT`, `--warmup-tokens N`, `--no-warmup` — startup warmup control.
- `--quiet` — suppress per-request access logs.

## Ollama-style API

```bash
curl http://127.0.0.1:11435/health
curl http://127.0.0.1:11435/api/tags
curl http://127.0.0.1:11435/api/version
```

Non-streaming completion:

```bash
curl http://127.0.0.1:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"motifcl-local:latest","prompt":"Hello","stream":false,"options":{"num_predict":32}}'
```

Streaming-shaped completion:

```bash
curl http://127.0.0.1:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"motifcl-local:latest","prompt":"Hello","stream":true,"options":{"num_predict":32}}'
```

Chat:

```bash
curl http://127.0.0.1:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"motifcl-local:latest","stream":false,"messages":[{"role":"user","content":"Say hi"}]}'
```

The server also exposes basic OpenAI-compatible endpoints for simple clients:

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

## Request options

The server accepts Ollama-style options either at top level or inside `options`:

- `num_predict` / `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `seed`
- `ignore_eos`

Internally each request is sent as one JSONL line to the persistent worker, so model weights and OpenCL runtime
state remain hot across requests. Responses include `worker_total_ms` from the MotifCL worker and HTTP-side
`total_duration` in nanoseconds.

## Current limits

- Streaming is API-compatible NDJSON chunking after the backend finishes the request. It is not true token-latency
  streaming yet; that requires a token callback inside the generation loop.
- `/api/chat` uses a simple role-prefixed prompt bridge. Exact per-model chat templates still belong in the
  runner arguments or a future template-aware server layer.
- The server owns one worker process and serializes requests through it. This matches current MotifCL single-GPU
  inference constraints and avoids concurrent writes to one REPL stdin/stdout pair.
- There is no Ollama-style model pull/push registry or daemon installer yet.

## Direct JSONL REPL

For low-overhead harnesses without HTTP:

```bash
printf '{"id":"smoke","prompt":"Hello","num_predict":8,"temperature":0}\n' | \
  ./build/tools/motifcl_generate_transformer --model ./model.gguf --jsonl-repl --completion-only
```

Each input line may be plain text or JSON with `prompt`/`input`, optional `id`, and the generation options above.
The JSONL response is one line:

```json
{"ok":true,"id":"smoke","response":"...","total_ms":12.345}
```
