#!/usr/bin/env python3
"""Compare motifcl_kernel_tuner JSON output against a stored baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def numeric_metrics(data: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path, help="current motifcl_tuning.json")
    parser.add_argument("baseline", type=Path, help="stored baseline JSON")
    parser.add_argument("--tolerance", type=float, default=0.15, help="allowed relative slowdown")
    parser.add_argument("--allow-missing", action="store_true", help="ignore metrics missing from current run")
    args = parser.parse_args()

    current = numeric_metrics(json.loads(args.current.read_text()))
    baseline = numeric_metrics(json.loads(args.baseline.read_text()))
    failures: list[str] = []
    for key, base_value in baseline.items():
        if key not in current:
            if not args.allow_missing:
                failures.append(f"{key}: missing from current run")
            continue
        cur_value = current[key]
        if cur_value > base_value * (1.0 + args.tolerance):
            failures.append(f"{key}: {cur_value:.4f} ms vs baseline {base_value:.4f} ms")

    if failures:
        print("Performance regressions detected:")
        for failure in failures:
            print(" -", failure)
        return 1
    print(f"Performance regression check passed for {len(baseline)} baseline metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
