#!/usr/bin/env python3
"""Optional parity harness for MotifCL's modern HF model path.

The harness is intentionally opt-in because it may need network access and the
optional ``torch``/``transformers`` stack:

    MOTIFCL_RUN_PARITY=1 PYTHONPATH=build/python python tools/parity_harness.py --repo hf-internal-testing/tiny-random-LlamaForCausalLM

It checks:
  * tokenizer encode/decode round trip on the MotifCL side;
  * prompt/chat-template rendering on the MotifCL side;
  * MotifCL logits shape and finite values after safetensors load;
  * optional HF logits comparison when torch/transformers can load the same repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tracemalloc
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

DEFAULT_FIXTURES = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "onnx-internal-testing/tiny-random-Qwen2ForCausalLM",
    "Xenova/tiny-random-GemmaForCausalLM",
]

TOKENIZER_CANDIDATES = [
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def urlopen_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "motifcl-parity-harness"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "motifcl-parity-harness"})
    with urllib.request.urlopen(req, timeout=120) as response, dst.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def hf_url(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


def sibling_size(info: dict, name: str) -> int | None:
    for item in info.get("siblings", []):
        if item.get("rfilename") == name and isinstance(item.get("size"), int):
            return int(item["size"])
    return None


def download_repo_subset(repo: str, root: Path, max_mb: int) -> tuple[Path, list[Path]]:
    info = urlopen_json(f"https://huggingface.co/api/models/{repo}")
    names = {item.get("rfilename", "") for item in info.get("siblings", []) if item.get("rfilename")}
    repo_dir = root / repo.replace("/", "__")
    repo_dir.mkdir(parents=True, exist_ok=True)
    if "config.json" not in names:
        raise RuntimeError(f"{repo}: config.json not found")
    download(hf_url(repo, "config.json"), repo_dir / "config.json")
    for name in TOKENIZER_CANDIDATES:
        if name in names:
            download(hf_url(repo, name), repo_dir / name)
    weights: list[Path] = []
    budget = max_mb * 1024 * 1024
    used = 0
    for name in sorted(n for n in names if n.endswith(".safetensors") and not n.endswith(".index.json")):
        size = sibling_size(info, name)
        if size is not None and used + size > budget:
            continue
        dst = repo_dir / name
        download(hf_url(repo, name), dst)
        used += dst.stat().st_size
        weights.append(dst)
    return repo_dir, weights


def motifcl_logits(repo_dir: Path, weights: list[Path], prompt: str) -> tuple[list[float], tuple[int, ...], list[int], str, float, int]:
    import motifcl as mcl

    backend = mcl.Backend.create()
    cfg = mcl.load_hf_transformer_config_json(str(repo_dir / "config.json"))
    spec = mcl.modern_model_spec_from_config(cfg)
    if not mcl.modern_model_spec_runnable_by_modern_gpt(spec):
        raise RuntimeError("model graph is not runnable by ModernGPTModel: " + "; ".join(spec.blockers))
    model = mcl.make_hf_transformer_model(backend, cfg)
    if weights:
        report = mcl.load_hf_transformer_weights(backend, model, [str(p) for p in weights], cfg, strict=False)
        if report.loaded_tensors <= 0:
            raise RuntimeError("no tensors loaded from safetensors")
    tokenizer = mcl.load_hf_tokenizer(str(repo_dir), cfg)
    ids = tokenizer.encode(prompt, False, False)
    if not ids:
        raise RuntimeError("MotifCL tokenizer produced no ids")
    messages = [mcl.HFChatMessage("system", "Be concise"), mcl.HFChatMessage("user", prompt)]
    template_kind = mcl.infer_hf_chat_template_kind(cfg.architecture, str(repo_dir))
    chat = mcl.apply_hf_chat_template(messages, cfg.architecture, template_kind, True)
    tokens = mcl.tensor_i32(backend, [1, len(ids)], ids)
    tracemalloc.start()
    t0 = time.perf_counter()
    logits = model.forward(tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    values = logits.cpu()
    if not values or any(not math.isfinite(v) for v in values):
        raise RuntimeError("MotifCL logits are empty or non-finite")
    # Exercise inference quantization on the same fixed fixture as a packed-quant smoke.
    # Native GGUF Q4_K/Q5_K loading is covered by the C++ GGUF fixture test; this keeps
    # the online harness independent of large remote GGUF artifacts.
    for qdtype in (mcl.DType.Q8_0, mcl.DType.Q4_0):
        mcl.enable_hf_transformer_quantized_inference(model, qdtype)
        q_logits = model.forward(tokens)
        q_values = q_logits.cpu()
        if tuple(q_logits.shape) != tuple(logits.shape) or any(not math.isfinite(v) for v in q_values):
            raise RuntimeError(f"packed quant smoke failed for {qdtype}")
    mcl.disable_hf_transformer_quantized_inference(model)
    return values, tuple(logits.shape), ids, chat, elapsed_ms, peak


def optional_hf_logits(repo_dir: Path, prompt: str) -> tuple[list[float], tuple[int, ...], list[int], str | None] | None:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception:
        return None
    tokenizer = AutoTokenizer.from_pretrained(repo_dir)
    model = AutoModelForCausalLM.from_pretrained(repo_dir, torch_dtype=torch.float32)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu().float()
    chat = None
    try:
        chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Be concise"}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        chat = None
    return logits.flatten().tolist(), tuple(int(x) for x in logits.shape), inputs["input_ids"].flatten().tolist(), chat


def compare_prefix(left: list[float], right: list[float], atol: float, rtol: float) -> float:
    n = min(len(left), len(right))
    if n == 0:
        raise RuntimeError("empty logits for comparison")
    return max(abs(left[i] - right[i]) / max(atol, abs(right[i]) * rtol) for i in range(n))


def compare_chat_template(motifcl_chat: str, hf_chat: str | None, repo: str) -> None:
    if hf_chat is None:
        print(f"{repo}: HF chat-template unavailable; MotifCL rendered {len(motifcl_chat)} chars")
        return
    if motifcl_chat != hf_chat:
        raise RuntimeError(f"{repo}: chat-template parity failed")
    print(f"{repo}: chat-template parity ok chars={len(motifcl_chat)}")


def run_repo(repo: str, cache_dir: Path, max_mb: int, prompt: str, atol: float, rtol: float) -> None:
    repo_dir, weights = download_repo_subset(repo, cache_dir, max_mb)
    m_logits, m_shape, m_ids, m_chat, elapsed_ms, peak = motifcl_logits(repo_dir, weights, prompt)
    print(f"{repo}: motifcl logits shape={m_shape} ids={m_ids[:16]} weights={len(weights)} latency_ms={elapsed_ms:.3f} peak_py_bytes={peak}")
    hf = optional_hf_logits(repo_dir, prompt)
    if hf is None:
        print(f"{repo}: transformers/torch unavailable; HF numerical parity skipped")
        return
    hf_logits, hf_shape, hf_ids, hf_chat = hf
    if m_ids != hf_ids:
        raise RuntimeError(f"{repo}: tokenizer parity failed motifcl={m_ids[:16]} hf={hf_ids[:16]}")
    print(f"{repo}: tokenizer parity ok tokens={len(m_ids)}")
    compare_chat_template(m_chat, hf_chat, repo)
    score = compare_prefix(m_logits, hf_logits, atol, rtol)
    print(f"{repo}: hf logits shape={hf_shape} normalized_max_error={score:.6g}")
    if score > 1.0:
        raise RuntimeError(f"{repo}: logits parity failed normalized_max_error={score:.6g}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="actually hit huggingface.co; otherwise skip")
    parser.add_argument("--repo", action="append", default=[], help="HF repo id; may be repeated")
    parser.add_argument("--cache-dir", type=Path, default=Path("build") / "parity_harness")
    parser.add_argument("--max-mb", type=int, default=64)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-2)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.run and os.environ.get("MOTIFCL_RUN_PARITY") != "1":
        print("Parity harness skipped; set MOTIFCL_RUN_PARITY=1 or pass --run")
        return 0
    repos = args.repo or DEFAULT_FIXTURES
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for repo in repos:
        try:
            run_repo(repo, args.cache_dir, args.max_mb, args.prompt, args.atol, args.rtol)
        except (urllib.error.URLError, RuntimeError, OSError, Exception) as exc:  # noqa: BLE001
            failures.append(f"{repo}: {exc}")
            print(f"{repo}: FAILED: {exc}", file=sys.stderr)
    if failures:
        for failure in failures:
            print(" -", failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
