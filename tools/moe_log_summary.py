#!/usr/bin/env python3
"""Summarize MotifCL pretrain logs (MoE and dense).

Parses the stdout logs of `motifcl_train_moe15m` / `motifcl_train_ternary15m`
and prints, per run: step, train loss, val loss and its perplexity, throughput,
device-memory peak and (for MoE) expert-load spread.

Usage:
    python tools/moe_log_summary.py out/moe_long/log.txt out/dense_long/log.txt
    python tools/moe_log_summary.py --json out/*/log.txt
    python tools/moe_log_summary.py --compare out/moe_long/log.txt out/moe_long/log_resume.txt out/dense_h384/log.txt

Missing files, missing val lines and unfinished runs are reported as such; the
script never invents a value for an absent measurement. In `--compare` mode logs
that live in the same directory are merged into one column (a resumed run split
across two files is one arm), and EVAL_ONLY metric lines are reported too.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

STEP_RE = re.compile(
    r"^step (?P<step>\d+) loss (?P<loss>[-\d.eE+]+) ema (?P<ema>[-\d.eE+]+) "
    r"lr (?P<lr>[-\d.eE+]+)\s+(?P<toks>[\d.]+) tok/s\s+eta [\d.]+ h"
    r"(?:\s+devmem (?P<mem_live>[\d.]+)/(?P<mem_peak>[\d.]+) MB)?"
    r"(?:\s+load (?P<load>[-\d. ]+?)(?=\s*spread|\s*bias0|$))?"
    r"(?:\s+spread (?P<spread>[\d.]+))?"
    r"(?:\s+bias0\[(?P<bias>[^\]]+)\])?"
    r"(?:\s+spread0 (?P<spread0>[\d.]+) gap0 (?P<gap0>[\d.]+))?"
)
VAL_RE = re.compile(r"^step (?P<step>\d+) VAL loss (?P<val>[-\d.eE+]+)")
FINAL_RE = re.compile(r"^FINAL VAL loss (?P<val>[-\d.eE+]+)")
EVALONLY_RE = re.compile(
    r"^EVAL_ONLY steps=(?P<step>\d+) records=(?P<records>\d+) tokens=(?P<tokens>\d+) "
    r"loss=(?P<loss>[-\d.eE+]+) ppl=(?P<ppl>[-\d.eE+]+) "
    r"top1=(?P<top1>[-\d.eE+]+) top5=(?P<top5>[-\d.eE+]+)"
)
HEADER_RE = re.compile(r"^(?P<name>\S+) pretrain: (?P<rest>.*)$")


def parse(path: Path) -> dict:
    run: dict = {
        "path": str(path),
        "exists": path.exists(),
        "name": None,
        "header": None,
        "steps": 0,
        "last_step": None,
        "loss": None,
        "ema": None,
        "val": [],
        "final_val": None,
        "tok_s": [],
        "mem_peak_mb": None,
        "spread": [],
        "load_last": None,
        "bytes": 0,
        "eval_only": None,
    }
    if not path.exists():
        return run
    run["bytes"] = path.stat().st_size
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = HEADER_RE.match(line)
        if header:
            run["name"] = header.group("name")
            run["header"] = header.group("rest")
            continue
        final = FINAL_RE.match(line)
        if final:
            run["final_val"] = float(final.group("val"))
            continue
        val = VAL_RE.match(line)
        if val:
            run["val"].append((int(val.group("step")), float(val.group("val"))))
            continue
        ev = EVALONLY_RE.match(line)
        if ev:
            run["eval_only"] = {
                "step": int(ev.group("step")),
                "loss": float(ev.group("loss")),
                "ppl": float(ev.group("ppl")),
                "top1": float(ev.group("top1")),
                "top5": float(ev.group("top5")),
            }
            continue
        step = STEP_RE.match(line)
        if step:
            run["steps"] += 1
            run["last_step"] = int(step.group("step"))
            run["loss"] = float(step.group("loss"))
            run["ema"] = float(step.group("ema"))
            run["tok_s"].append(float(step.group("toks")))
            if step.group("mem_peak"):
                run["mem_peak_mb"] = float(step.group("mem_peak"))
            if step.group("spread"):
                run["spread"].append(float(step.group("spread")))
            if step.group("load"):
                try:
                    run["load_last"] = [float(x) for x in step.group("load").split()]
                except ValueError:
                    pass
    return run


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ppl(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def fmt(value, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def compare(runs: list[dict]) -> int:
    """Merge runs by parent directory and print their val curves side by side."""
    arms: dict[str, dict] = {}
    for run in runs:
        label = Path(run["path"]).parent.name or run["path"]
        arm = arms.setdefault(label, {"val": {}, "final_val": None, "tok_s": [], "mem_peak_mb": None,
                                      "eval_only": None, "missing": []})
        if not run["exists"]:
            arm["missing"].append(run["path"])
            continue
        for step, value in run["val"]:
            arm["val"][step] = value
        if run["final_val"] is not None:
            arm["final_val"] = run["final_val"]
        arm["tok_s"].extend(run["tok_s"])
        if run["mem_peak_mb"] is not None:
            arm["mem_peak_mb"] = max(arm["mem_peak_mb"] or 0.0, run["mem_peak_mb"])
        if run["eval_only"] is not None:
            arm["eval_only"] = run["eval_only"]

    steps = sorted({s for arm in arms.values() for s in arm["val"]})
    if not steps:
        print("no val points found in any log")
        return 1
    labels = list(arms)
    width = max(6, max(len(x) for x in labels) + 1)
    print("step".ljust(8) + "".join(x.ljust(width) for x in labels))
    for step in steps:
        row = str(step).ljust(8)
        for label in labels:
            value = arms[label]["val"].get(step)
            row += (fmt(value) if value is not None else "-").ljust(width)
        print(row)
    print()
    for label in labels:
        arm = arms[label]
        best = None
        for step in steps:
            value = arm["val"].get(step)
            if value is not None:
                best = (step, value)
        print(
            f"{label}: last val {fmt(best[1]) if best else 'n/a'} "
            f"(step {best[0] if best else '-'}), final {fmt(arm['final_val'])}, "
            f"ppl {fmt(ppl(arm['final_val'] if arm['final_val'] is not None else (best[1] if best else None)), '.3f')}, "
            f"tok/s mean {fmt(mean(arm['tok_s']), '.0f')}, device peak {fmt(arm['mem_peak_mb'], '.0f')} MB"
        )
        if arm["eval_only"]:
            e = arm["eval_only"]
            print(
                f"    EVAL_ONLY step={e['step']}: loss={e['loss']:.4f} ppl={e['ppl']:.3f} "
                f"top1={e['top1']:.4f} top5={e['top5']:.4f}"
            )
        for path in arm["missing"]:
            print(f"    missing: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--compare", action="store_true",
                    help="merge logs by directory and print their val curves side by side")
    args = ap.parse_args()

    runs = [parse(Path(p)) for p in args.logs]
    if args.compare:
        return compare(runs)
    if args.json:
        out = []
        for run in runs:
            out.append(
                {
                    "path": run["path"],
                    "exists": run["exists"],
                    "last_step": run["last_step"],
                    "train_loss": run["loss"],
                    "val": run["val"],
                    "final_val": run["final_val"],
                    "val_ppl": ppl(run["final_val"] if run["final_val"] is not None else (run["val"][-1][1] if run["val"] else None)),
                    "tok_s_last": run["tok_s"][-1] if run["tok_s"] else None,
                    "tok_s_mean": mean(run["tok_s"]),
                    "mem_peak_mb": run["mem_peak_mb"],
                    "spread_last": run["spread"][-1] if run["spread"] else None,
                    "spread_min": min(run["spread"]) if run["spread"] else None,
                    "load_last": run["load_last"],
                }
            )
        print(json.dumps(out, indent=2))
        return 0

    for run in runs:
        print(f"== {run['path']}")
        if not run["exists"]:
            print("   file does not exist")
            continue
        if run["name"]:
            print(f"   {run['name']}: {run['header']}")
        print(
            f"   log lines={run['steps']}  last step={run['last_step']}  "
            f"train loss={fmt(run['loss'])}  ema={fmt(run['ema'])}"
        )
        last_val = run["final_val"] if run["final_val"] is not None else (run["val"][-1][1] if run["val"] else None)
        print(f"   val last={fmt(last_val)}  ppl={fmt(ppl(last_val), '.3f')}  val points={len(run['val'])}")
        if run["val"]:
            print("   val curve: " + "  ".join(f"{s}:{v:.4f}" for s, v in run["val"]))
        print(
            f"   tok/s last={fmt(run['tok_s'][-1] if run['tok_s'] else None, '.0f')}  "
            f"mean={fmt(mean(run['tok_s']), '.0f')}  device peak={fmt(run['mem_peak_mb'], '.0f')} MB"
        )
        if run["spread"]:
            print(
                f"   expert load spread last={run['spread'][-1]:.3f} "
                f"min={min(run['spread']):.3f}"
            )
        if run["load_last"]:
            print("   last load: " + " ".join(f"{x:.3f}" for x in run["load_last"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
