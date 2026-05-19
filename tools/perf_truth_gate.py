#!/usr/bin/env python3
"""Strict wrapper for measured fast-path performance gates.

Every optional native/ASM/OpenCL fast path should land with:
  1. correctness tests in CTest/pytest;
  2. a baseline artifact;
  3. a candidate artifact;
  4. this gate proving the candidate is not slower beyond tolerance.

The script intentionally delegates metric comparison to perf_regression.py and
adds fast-path policy language around it, so CI/local gates have one stable
entry point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="MotifCL fast-path performance truth gate")
    parser.add_argument("baseline", help="baseline JSON emitted by the stable/default path")
    parser.add_argument("candidate", help="candidate JSON emitted by the new fast path")
    parser.add_argument("--tolerance", default="0.15", help="allowed regression ratio; default 0.15")
    parser.add_argument("--min-metrics", default="1")
    parser.add_argument("--require-key", action="append", default=[])
    parser.add_argument("--require-field", action="append", default=[])
    parser.add_argument("--require-baseline-field", action="append", default=[])
    args = parser.parse_args()

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    if not baseline.exists():
        raise SystemExit(f"baseline artifact not found: {baseline}")
    if not candidate.exists():
        raise SystemExit(f"candidate artifact not found: {candidate}")

    cmd = [
        sys.executable,
        "tools/perf_regression.py",
        str(baseline),
        str(candidate),
        "--tolerance",
        str(args.tolerance),
        "--min-metrics",
        str(args.min_metrics),
    ]
    for key in args.require_key:
        cmd += ["--require-key", key]
    for field in args.require_field:
        cmd += ["--require-field", field]
    for field in args.require_baseline_field:
        cmd += ["--require-baseline-field", field]
    run(cmd)
    print("Performance truth gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
