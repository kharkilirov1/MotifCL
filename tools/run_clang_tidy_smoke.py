#!/usr/bin/env python3
"""Small CI-oriented clang-tidy diagnostics gate.

This intentionally checks a tiny stable translation-unit set with compiler
diagnostics/analyzer checks. It is not a style-enforcement pass over the whole
tree; use it to catch broken compile commands and clang diagnostics in CI.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("files", nargs="*", default=["src/core/shape.cpp", "src/core/error.cpp"])
    args = parser.parse_args()

    clang_tidy = shutil.which("clang-tidy")
    if clang_tidy is None:
        print("clang-tidy not found; skipping smoke gate")
        return 0

    build_dir = args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir
    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.exists():
        raise SystemExit(f"missing {compile_commands}; configure with CMAKE_EXPORT_COMPILE_COMMANDS=ON")

    files = [
        str((Path(file) if Path(file).is_absolute() else ROOT / file).resolve())
        for file in args.files
    ]
    # Keep the smoke gate resilient when clang-tidy cannot match a source file
    # to the compile database on a hosted runner: the checked translation units
    # only need the public include tree and bundled OpenCL headers.
    extra_args = [
        f"-I{ROOT / 'include'}",
        "-isystem",
        str(ROOT / "third_party" / "opencl"),
        "-DCL_TARGET_OPENCL_VERSION=120",
        "-std=c++17",
    ]

    cmd = [
        clang_tidy,
        "-p",
        str(build_dir),
        "--checks=clang-diagnostic-*,clang-analyzer-*",
        "--warnings-as-errors=clang-diagnostic-*",
        "--quiet",
        *files,
        "--",
        *extra_args,
    ]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
