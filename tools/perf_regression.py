#!/usr/bin/env python3
"""Compare motifcl_kernel_tuner JSON output against a stored baseline.

The comparison is intentionally JSON-only so it can be used for local release
checks and CI artifact gates. Numeric top-level values are treated as timing
metrics. Optional metadata gates can check raw top-level keys or dotted fields
inside a ``metadata`` object without affecting the timing comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_MISSING = object()


def numeric_metrics(data: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def field_value(data: dict[str, Any], field: str) -> object:
    """Return a raw JSON field by exact key, dotted path, or metadata fallback."""
    if field in data:
        return data[field]
    node: object = data
    for part in field.split("."):
        if not isinstance(node, dict) or part not in node:
            node = _MISSING
            break
        node = node[part]
    if node is not _MISSING:
        return node
    metadata = data.get("metadata")
    if "." not in field and isinstance(metadata, dict) and field in metadata:
        return metadata[field]
    return _MISSING


def field_string(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path, help="current motifcl_tuning.json")
    parser.add_argument("baseline", type=Path, help="stored baseline JSON")
    parser.add_argument("--tolerance", type=float, default=0.15, help="allowed relative slowdown")
    parser.add_argument("--allow-missing", action="store_true", help="ignore metrics missing from current run")
    parser.add_argument("--min-metrics", type=int, default=1, help="minimum numeric metrics required in current run")
    parser.add_argument("--require-key", action="append", default=[], help="metric key that must be present; may be repeated")
    parser.add_argument("--require-current-key", action="append", default=[], help="alias for --require-key")
    parser.add_argument("--require-baseline-key", action="append", default=[], help="baseline metric key that must be present")
    parser.add_argument("--require-field", action="append", default=[], help="raw current JSON field/dotted path that must exist")
    parser.add_argument("--require-baseline-field", action="append", default=[], help="raw baseline JSON field/dotted path that must exist")
    parser.add_argument("--match-field", action="append", default=[], help="raw field/dotted path that must match current vs baseline")
    parser.add_argument("--require-field-contains", nargs=2, action="append", default=[], metavar=("FIELD", "TEXT"),
                        help="raw current field/dotted path that must contain TEXT")
    parser.add_argument("--print-metadata", action="store_true", help="print current/baseline metadata objects when present")
    args = parser.parse_args()

    current_raw = json.loads(args.current.read_text())
    baseline_raw = json.loads(args.baseline.read_text())
    current = numeric_metrics(current_raw)
    baseline = numeric_metrics(baseline_raw)
    failures: list[str] = []
    if len(current) < args.min_metrics:
        failures.append(f"current run has {len(current)} numeric metrics, expected at least {args.min_metrics}")
    for key in args.require_key + args.require_current_key:
        if key not in current:
            failures.append(f"{key}: missing required current metric")
    for key in args.require_baseline_key:
        if key not in baseline:
            failures.append(f"{key}: missing required baseline metric")
    for field in args.require_field:
        if field_value(current_raw, field) is _MISSING:
            failures.append(f"{field}: missing required current field")
    for field in args.require_baseline_field:
        if field_value(baseline_raw, field) is _MISSING:
            failures.append(f"{field}: missing required baseline field")
    for field in args.match_field:
        cur = field_value(current_raw, field)
        base = field_value(baseline_raw, field)
        if cur is _MISSING or base is _MISSING:
            failures.append(f"{field}: missing field required for metadata match")
        elif cur != base:
            failures.append(f"{field}: current {field_string(cur)!r} != baseline {field_string(base)!r}")
    for field, needle in args.require_field_contains:
        cur = field_value(current_raw, field)
        if cur is _MISSING:
            failures.append(f"{field}: missing field required to contain {needle!r}")
        elif needle not in field_string(cur):
            failures.append(f"{field}: {field_string(cur)!r} does not contain {needle!r}")
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
    if args.print_metadata:
        if isinstance(current_raw.get("metadata"), dict):
            print("Current metadata:", json.dumps(current_raw["metadata"], sort_keys=True))
        if isinstance(baseline_raw.get("metadata"), dict):
            print("Baseline metadata:", json.dumps(baseline_raw["metadata"], sort_keys=True))
    print(f"Performance regression check passed for {len(baseline)} baseline metrics; current metrics={len(current)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
