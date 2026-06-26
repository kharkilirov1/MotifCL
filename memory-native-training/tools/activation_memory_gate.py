#!/usr/bin/env python3
"""Activation-memory frontier gate.

The existing memory_truth_gate verifies that dense decoded weights and full grad_w do not reappear.
This script checks the next frontier: whether the counter backward node still stores the full input
activation Tensor x.  By default it reports a warning and exits 0, because non-reversible training
legitimately needs x.  Use --strict when testing a reversible implementation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "engine" / "src" / "compact_counter.cpp"


def extract_struct(text: str, name: str) -> str:
    m = re.search(rf"struct\s+{re.escape(name)}\s*:[\s\S]*?\n\}};", text)
    if not m:
        raise AssertionError(f"could not find struct {name}")
    return m.group(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    cpp = (args.root / CPP.relative_to(ROOT)).read_text(encoding="utf-8")
    node = extract_struct(cpp, "CounterBackwardNode")
    stores_x = bool(re.search(r"\bTensor\s+x\s*;", node))
    if stores_x:
        msg = ("ACTIVATION FRONTIER: CounterBackwardNode still stores Tensor x. "
               "This is expected for ordinary training, but a reversible/quantized-activation path "
               "must remove or replace it.")
        if args.strict:
            raise AssertionError(msg)
        print("WARN " + msg)
    else:
        print("PASS activation-memory gate: CounterBackwardNode does not store full Tensor x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
