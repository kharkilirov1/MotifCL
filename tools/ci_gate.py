#!/usr/bin/env python3
"""MotifCL local CI gate: release build, full CTest, Python build/tests, and release_check."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = str(ROOT / "build" / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = py_path if not existing else py_path + os.pathsep + existing
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"))
    parser.add_argument("--skip-release-check", action="store_true")
    parser.add_argument("--release-check-extra", action="append", default=[],
                        help="extra argument passed through to tools/release_check.py")
    args = parser.parse_args()

    run(["git", "diff", "--check"])
    run([sys.executable, "tools/check_kernel_contracts.py"])
    run(["cmake", "--preset", "release"])
    run(["cmake", "--build", "--preset", "release", "--parallel", args.parallel])
    run(["ctest", "--test-dir", "build/release", "--output-on-failure"])
    run(["cmake", "--preset", "python"])
    run(["cmake", "--build", "--preset", "python", "--parallel", args.parallel])
    run([sys.executable, "-m", "pytest", "tests/python", "-q"], env=python_env())
    run([sys.executable, "-m", "py_compile",
         "tools/ci_gate.py", "tools/check_kernel_contracts.py",
         "tools/perf_truth_gate.py", "tools/release_check.py",
         "tools/motifcl_cli.py", "tools/motifcl_ollama_server.py"])
    if not args.skip_release_check:
        run([sys.executable, "tools/release_check.py", "--parallel", args.parallel, *args.release_check_extra])
    print("CI gate completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
