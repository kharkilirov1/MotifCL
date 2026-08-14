#!/usr/bin/env python3
"""Prepare a flat int32 token stream for MotifCL FOG Vulkan training.

Accepted input:
  * JSONL with a text field (default: text)
  * TXT, one document per line
  * --tinystories N: download the first N TinyStories rows via datasets

The output is raw little-endian int32 token ids. Documents are separated by
<|endoftext|> when the tokenizer exposes it, otherwise by the tokenizer EOS id
if present, otherwise no explicit separator is inserted.
"""
from __future__ import annotations
import argparse, json, struct
from pathlib import Path


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install tokenizers") from exc
    return Tokenizer.from_file(str(path))


def iter_texts(path: Path, field: str):
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(field)
                if isinstance(text, str) and text.strip():
                    yield text
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def iter_tinystories(limit: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("TinyStories download needs: pip install datasets") from exc
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="ports/rx580/fog/tinystories_3k_bpe.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input")
    src.add_argument("--tinystories", type=int)
    ap.add_argument("--field", default="text")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()

    tok = load_tokenizer(Path(args.tokenizer))
    sep = None
    vocab = tok.get_vocab()
    for candidate in ("<|endoftext|>", "</s>", "<eos>"):
        if candidate in vocab:
            sep = vocab[candidate]
            break

    texts = iter_tinystories(args.tinystories) if args.tinystories else iter_texts(Path(args.input), args.field)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    tokens = 0
    with out_path.open("wb") as f:
        for text in texts:
            if args.max_docs and docs >= args.max_docs:
                break
            ids = tok.encode(text).ids
            if sep is not None:
                ids.append(sep)
            if ids:
                f.write(struct.pack("<" + "i" * len(ids), *ids))
                tokens += len(ids)
                docs += 1
            if docs and docs % 1000 == 0:
                print(f"docs={docs} tokens={tokens}", flush=True)
    print(json.dumps({"output": str(out_path), "documents": docs, "tokens": tokens, "separator_id": sep}, ensure_ascii=False))


if __name__ == "__main__":
    main()
