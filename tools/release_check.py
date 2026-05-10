#!/usr/bin/env python3
"""Run MotifCL release-readiness checks without mutating git state.

The script orchestrates the same checks documented for local release cleanup:
CMake dev/release/python builds, CTest, Python API tests, optional HF smoke,
perf-gate tooling, install/consumer smoke, and optional wheel smoke in a
temporary virtual environment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
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


def build_and_test_dev(parallel: str) -> None:
    run(["cmake", "--preset", "dev"])
    run(["cmake", "--build", "--preset", "dev", "--parallel", parallel])
    run(["ctest", "--preset", "dev", "--output-on-failure"])


def build_release(parallel: str) -> None:
    run(["cmake", "--preset", "release"])
    run(["cmake", "--build", "--preset", "release", "--parallel", parallel])


def build_and_test_python(parallel: str, *, hf_run: bool) -> None:
    run(["cmake", "--preset", "python"])
    run(["cmake", "--build", "--preset", "python", "--parallel", parallel])
    env = python_env()
    run([sys.executable, "-m", "pytest", "tests/python", "-q"], env=env)
    hf_cmd = [sys.executable, "tools/hf_integration_smoke.py"]
    if hf_run:
        hf_cmd.append("--run")
    run(hf_cmd, env=env)


def run_perf_gate() -> None:
    run([
        sys.executable,
        "tools/perf_regression.py",
        "baselines/ellesmere.json",
        "baselines/ellesmere.json",
        "--tolerance",
        "0.15",
        "--min-metrics",
        "9",
        "--require-key",
        "matmul_q8_256_ms",
        "--require-key",
        "matmul_q4_256_ms",
        "--require-field",
        "metadata.device_family",
        "--require-baseline-field",
        "metadata.profile",
    ])


def run_static_cleanup_checks() -> None:
    run([sys.executable, "-m", "compileall", "-q", "python/motifcl"])
    run(["git", "diff", "--check"])


def run_consumer_smoke(parallel: str) -> None:
    prefix = ROOT / "build" / "install-smoke"
    run(["cmake", "--install", str(ROOT / "build" / "dev"), "--prefix", str(prefix)])
    run(["cmake", "-S", "tests/consumer", "-B", "build/consumer-smoke", f"-DCMAKE_PREFIX_PATH={prefix}"])
    run(["cmake", "--build", "build/consumer-smoke", "--parallel", parallel])
    exe = ROOT / "build" / "consumer-smoke" / ("motifcl_consumer_smoke.exe" if os.name == "nt" else "motifcl_consumer_smoke")
    run([str(exe)])


def run_wheel_smoke() -> None:
    run([sys.executable, "-m", "build", "--wheel"])
    wheels = sorted((ROOT / "dist").glob("motifcl-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError("wheel build completed but no dist/motifcl-*.whl was found")
    wheel = wheels[-1]
    with tempfile.TemporaryDirectory(prefix="motifcl-wheel-smoke-") as tmp:
        venv = Path(tmp) / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(py), "-m", "pip", "install", "--upgrade", "pip", "pytest"])
        run([str(py), "-m", "pip", "install", str(wheel)])
        run([
            str(py),
            "-c",
            (
                "import motifcl as mcl; "
                "b=mcl.Backend.create(); "
                "x=mcl.ones(b,[2,3]); "
                "assert x.shape == [2,3]; "
                "print('wheel smoke passed')"
            ),
        ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"))
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-consumer", action="store_true")
    parser.add_argument("--hf-run", action="store_true", help="run real Hugging Face network smoke instead of offline skip")
    parser.add_argument("--wheel", action="store_true", help="also build and install the wheel in a temporary venv")
    parser.add_argument("--require-clean-git", action="store_true", help="fail if the working tree has local changes")
    args = parser.parse_args()

    if args.require_clean_git:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        if status.strip():
            print("Working tree is not clean:", file=sys.stderr)
            print(status, file=sys.stderr)
            return 1

    if not args.skip_dev:
        build_and_test_dev(args.parallel)
    if not args.skip_release:
        build_release(args.parallel)
    if not args.skip_python:
        build_and_test_python(args.parallel, hf_run=args.hf_run)
    if not args.skip_perf:
        run_perf_gate()
    if not args.skip_static:
        run_static_cleanup_checks()
    if not args.skip_consumer:
        run_consumer_smoke(args.parallel)
    if args.wheel:
        run_wheel_smoke()
    print("Release check completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
