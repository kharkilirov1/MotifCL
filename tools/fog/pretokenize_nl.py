#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-tokenize the JSONL stories from gen_nl_stories.py into a compact line
format for the C++ trainer/inference:

    <init> <d1,d2,...,dR> <id1,id2,...,idT>

Reads  data/<name>.jsonl  writes data/<name>.tok
"""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

TOK = Path("ports/rx580/fog/tinystories_3k_bpe.json")

def main():
    tok = Tokenizer.from_file(str(TOK))
    for name in ["fog_nl_train", "fog_nl_test"]:
        src = Path("data") / f"{name}.jsonl"
        dst = Path("data") / f"{name}.tok"
        n = 0
        with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as o:
            for line in f:
                line = line.strip()
                if not line: continue
                ex = json.loads(line)
                ids = tok.encode(ex["text"]).ids
                dcsv = ",".join(str(x) for x in ex["deltas"])
                tcsv = ",".join(str(x) for x in ids)
                o.write(f"{ex['init']} {dcsv} {tcsv}\n")
                n += 1
        print(f"wrote {n} -> {dst}")

if __name__ == "__main__":
    main()