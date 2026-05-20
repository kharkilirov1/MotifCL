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
    parser.add_argument("--min-q4-speedup", type=float, default=1.05)
    parser.add_argument("--max-abs-diff", type=float, default=1e-4)
    parser.add_argument("--require-simd", action="store_true")
    parser.add_argument("--require-q4", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    scalar_ms = float(data["native_matmul_f32_m1_scalar_ms"])
    dispatch_ms = float(data["native_matmul_f32_m1_dispatch_ms"])
    max_abs_diff = float(data["native_matmul_f32_m1_max_abs_diff"])
    metadata = data.get("metadata", {})
    kernel = str(metadata.get("kernel", "unknown"))
    q4_kernel = str(metadata.get("q4_kernel", "unknown"))
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
    q4_status = ""
    if args.require_q4 or "native_matmul_f32_q4_0_m1_scalar_ms" in data:
        q4_scalar_ms = float(data["native_matmul_f32_q4_0_m1_scalar_ms"])
        q4_dispatch_ms = float(data["native_matmul_f32_q4_0_m1_dispatch_ms"])
        q4_max_abs_diff = float(data["native_matmul_f32_q4_0_m1_max_abs_diff"])
        if (
            not math.isfinite(q4_scalar_ms)
            or not math.isfinite(q4_dispatch_ms)
            or q4_scalar_ms <= 0.0
            or q4_dispatch_ms <= 0.0
        ):
            raise SystemExit("invalid Q4 benchmark timings")
        if q4_max_abs_diff > args.max_abs_diff:
            raise SystemExit(
                f"native Q4 dispatch mismatch: max_abs_diff={q4_max_abs_diff} > {args.max_abs_diff}"
            )
        if args.require_simd and q4_kernel.startswith("scalar"):
            raise SystemExit("SIMD Q4 dispatch required but native Q4 kernel is scalar")
        q4_speedup = q4_scalar_ms / q4_dispatch_ms
        if q4_speedup < args.min_q4_speedup:
            raise SystemExit(
                f"native Q4 dispatch speedup {q4_speedup:.3f}x below required {args.min_q4_speedup:.3f}x "
                f"(scalar={q4_scalar_ms:.6f} ms dispatch={q4_dispatch_ms:.6f} ms kernel={q4_kernel})"
            )
        q4_status = (
            f"; Q4 {q4_speedup:.3f}x "
            f"(scalar={q4_scalar_ms:.6f} ms dispatch={q4_dispatch_ms:.6f} ms kernel={q4_kernel})"
        )
    print(
        f"Native matmul breakthrough gate passed: {speedup:.3f}x "
        f"(scalar={scalar_ms:.6f} ms dispatch={dispatch_ms:.6f} ms kernel={kernel})"
        f"{q4_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
