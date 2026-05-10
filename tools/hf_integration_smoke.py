#!/usr/bin/env python3
"""Optional real Hugging Face integration smoke for MotifCL's modern HF path.

Disabled by default to keep unit tests offline/reproducible. Run with either:

    MOTIFCL_RUN_HF_INTEGRATION=1 PYTHONPATH=build/python python tools/hf_integration_smoke.py

or:

    PYTHONPATH=build/python python tools/hf_integration_smoke.py --run --repo hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

DEFAULT_REPOS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "onnx-internal-testing/tiny-random-Qwen2ForCausalLM",
    "Xenova/tiny-random-GemmaForCausalLM",
]

TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def urlopen_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "motifcl-hf-integration-smoke"})
    with urllib.request.urlopen(req, timeout=120) as response, dst.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def sibling_names(info: dict) -> list[str]:
    return [item.get("rfilename", "") for item in info.get("siblings", []) if item.get("rfilename")]


def sibling_size(info: dict, name: str) -> int | None:
    for item in info.get("siblings", []):
        if item.get("rfilename") == name:
            size = item.get("size")
            return int(size) if isinstance(size, int) else None
    return None


def hf_resolve_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download_repo_subset(repo: str, root: Path, max_mb: int) -> tuple[Path, list[Path]]:
    info = urlopen_json(f"https://huggingface.co/api/models/{repo}")
    names = sibling_names(info)
    repo_dir = root / repo.replace("/", "__")
    repo_dir.mkdir(parents=True, exist_ok=True)

    if "config.json" not in names:
        raise RuntimeError(f"{repo}: config.json not found")
    download(hf_resolve_url(repo, "config.json"), repo_dir / "config.json")

    for name in TOKENIZER_CANDIDATES:
        if name in names:
            download(hf_resolve_url(repo, name), repo_dir / name)

    safetensors = [n for n in names if n.endswith(".safetensors") and not n.endswith(".index.json")]
    if not safetensors and "model.safetensors" in names:
        safetensors = ["model.safetensors"]
    downloaded_weights: list[Path] = []
    max_bytes = max_mb * 1024 * 1024
    total = 0
    for name in sorted(safetensors):
        size = sibling_size(info, name)
        if size is not None and total + size > max_bytes:
            continue
        dst = repo_dir / name
        download(hf_resolve_url(repo, name), dst)
        total += dst.stat().st_size
        downloaded_weights.append(dst)
    return repo_dir, downloaded_weights


def smoke_repo(repo: str, cache_dir: Path, max_mb: int, prompt: str) -> None:
    import motifcl as mcl

    repo_dir, weights = download_repo_subset(repo, cache_dir, max_mb)
    backend = mcl.Backend.create()
    cfg = mcl.load_hf_transformer_config_json(str(repo_dir / "config.json"))
    model = mcl.make_hf_transformer_model(backend, cfg)
    if weights:
        report = mcl.load_hf_transformer_weights(backend, model, [str(p) for p in weights], cfg, strict=False)
        if report.loaded_tensors <= 0:
            raise RuntimeError(f"{repo}: no tensors were loaded from safetensors")
        print(f"{repo}: loaded={report.loaded_tensors} missing={len(report.missing)} unexpected={len(report.unexpected)}")
    else:
        print(f"{repo}: no safetensors under {max_mb} MiB, using random weights for loader/tokenizer smoke")
    tokenizer = mcl.load_hf_tokenizer(str(repo_dir), cfg)
    opts = mcl.GenerateOptions()
    opts.max_new_tokens = 0
    opts.eos_token_id = cfg.eos_token_id
    opts.pad_token_id = cfg.pad_token_id
    opts.top_p = 0.9
    text = mcl.generate_hf_text(backend, model, tokenizer, prompt, opts)
    if not isinstance(text, str) or len(text) == 0:
        raise RuntimeError(f"{repo}: empty generated text")
    # Exercise quant policy on the real config without making this a quality test.
    policy = mcl.QuantizationPolicy()
    policy.default_dtype = mcl.DType.Q4_0
    policy.lm_head_dtype = mcl.DType.Q8_0
    if cfg.transformer.n_layer > 0:
        policy.q8_layers = [0]
    model.enable_quantized_inference_policy(policy)
    opts.max_new_tokens = 0
    _ = mcl.generate_hf_text(backend, model, tokenizer, prompt, opts)
    print(f"{repo}: ok arch={cfg.architecture_name} vocab={cfg.transformer.vocab_size}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="actually hit huggingface.co; otherwise skip")
    parser.add_argument("--repo", action="append", default=[], help="HF repo id; may be repeated")
    parser.add_argument("--cache-dir", type=Path, default=Path("build") / "hf_integration_smoke")
    parser.add_argument("--max-mb", type=int, default=64, help="max total safetensors MiB per repo")
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.run and os.environ.get("MOTIFCL_RUN_HF_INTEGRATION") != "1":
        print("HF integration smoke skipped; set MOTIFCL_RUN_HF_INTEGRATION=1 or pass --run")
        return 0

    repos = args.repo or DEFAULT_REPOS
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for repo in repos:
        try:
            smoke_repo(repo, args.cache_dir, args.max_mb, args.prompt)
        except (urllib.error.URLError, RuntimeError, OSError, Exception) as exc:  # noqa: BLE001: smoke should aggregate failures
            failures.append(f"{repo}: {exc}")
            print(f"{repo}: FAILED: {exc}", file=sys.stderr)
    if failures:
        print("HF integration smoke failures:", file=sys.stderr)
        for failure in failures:
            print(" -", failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
