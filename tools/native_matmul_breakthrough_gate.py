#!/usr/bin/env python3
"""Gate standalone native matmul dispatch against its scalar baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="MotifCL native matmul breakthrough gate")
    parser.add_argument("artifact", type=Path, help="JSON emitted by bench_native_matmul")
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--max-abs-diff", type=float, default=1e-4)
    parser.add_argument("--require-simd", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    scalar_ms = float(data["native_matmul_f32_m1_scalar_ms"])
    dispatch_ms = float(data["native_matmul_f32_m1_dispatch_ms"])
    max_abs_diff = float(data["native_matmul_f32_m1_max_abs_diff"])
    metadata = data.get("metadata", {})
    kernel = str(metadata.get("kernel", "unknown"))
    if not math.isfinite(scalar_ms) or not math.isfinite(dispatch_ms) or scalar_ms <= 0.0 or dispatch_ms <= 0.0:
        raise SystemExit("invalid benchmark timings")
    if max_abs_diff > args.max_abs_diff:
        raise SystemExit(f"native dispatch mismatch: max_abs_diff={max_abs_diff} > {args.max_abs_diff}")
    if args.require_simd and kernel == "scalar":
        raise SystemExit("SIMD dispatch required but native kernel is scalar")
    speedup = scalar_ms / dispatch_ms
    if speedup < args.min_speedup:
        raise SystemExit(
            f"native dispatch speedup {speedup:.3f}x below required {args.min_speedup:.3f}x "
            f"(scalar={scalar_ms:.6f} ms dispatch={dispatch_ms:.6f} ms kernel={kernel})"
        )
    print(
        f"Native matmul breakthrough gate passed: {speedup:.3f}x "
        f"(scalar={scalar_ms:.6f} ms dispatch={dispatch_ms:.6f} ms kernel={kernel})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
