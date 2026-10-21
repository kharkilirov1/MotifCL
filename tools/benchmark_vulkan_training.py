#!/usr/bin/env python3
"""Reproducible Vulkan training ABBA comparison, with immutable logs and hashes.

Example: python tools/benchmark_vulkan_training.py baseline.exe candidate.exe
  --data-dir data/tinystories --output-dir out/abba --steps 120 --pairs 2
--profile is separate: one run per executable, kernel profiling, no speedup.
The output directory must not exist. No ambient training/Vulkan overrides leak in.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ENV = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
              "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "COMSPEC", "PATHEXT",
              "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ"}
DEFAULTS = dict(BATCH="8", SEQ="256", SEED="1337", DIM="288", HEADS="6", LAYERS="6",
                HIDDEN_EXP="192", HIDDEN_SHARED="192", HIDDEN="768", TIE_HEAD="0",
                LR="0.004", LR_FP="0.006", WARMUP_FRAC="0.02", MIN_LR_FRAC="0.1",
                BALANCE_GAMMA="0.01", AUX_WEIGHT="0.01", BALANCE_CLAMP="2",
                EVAL_EVERY="100000", EVAL_RECORDS="32", EVAL_BATCH="8",
                CKPT_EVERY="100000", SAVE_CKPT="0", RESUME="0", EVAL_ONLY="0",
                ROUTE_HIST="0", GPU_TIMING="1", REC_LEN="257")
STEP_RE = re.compile(r"^step (\d+) loss (\S+) ema (\S+) lr (\S+)\s+(\S+) tok/s([^\n]*)", re.M)
FINAL_RE = re.compile(r"^FINAL VAL loss (\S+)", re.M)
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
LEGACY_STEP_JOIN_RE = re.compile(rf"(^[ \t]+ms gpu {FLOAT} sync {FLOAT})(?=step \d+ loss )", re.M)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_environment(parent, *, data_dir, run_dir, steps, log_every, profile):
    env = {key: value for key, value in parent.items() if key.upper() in SYSTEM_ENV}
    for key in list(env):
        if key.upper() == "PATH" and key != "PATH":
            env["PATH"] = env.pop(key)
    env.update(DEFAULTS)
    env.update(DATA_DIR=str(Path(data_dir).resolve()), OUT_DIR=str(Path(run_dir).resolve()),
               STEPS=str(steps), LOG_EVERY=str(log_every), MOTIFCL_VK_PROFILE="1" if profile else "0")
    dll_dir = Path("C:/Strawberry/c/bin")
    if os.name == "nt" and dll_dir.is_dir():
        env["PATH"] = str(dll_dir) + os.pathsep + env.get("PATH", "")
    return env


def parse_log(text, exit_code, *, steps, warmup=20, log_every=20, require_steady=True):
    # Immutable old dense binaries omitted the newline after their timing prefix.
    text = LEGACY_STEP_JOIN_RE.sub(r"\1\n", text)
    errors, windows, final_values = [], [], FINAL_RE.findall(text)
    if exit_code != 0:
        errors.append(f"process exit code {exit_code}")
    previous = 0
    for match in STEP_RE.finditer(text):
        step = int(match[1])
        try:
            loss, ema, lr, rate = map(float, match.group(2,3,4,5))
        except ValueError:
            loss = ema = lr = rate = float("nan")
        if not all(math.isfinite(value) for value in (loss,ema,lr,rate)) or rate <= 0:
            errors.append(f"non-finite training measurement or non-positive throughput at step {step}")
        explicit = re.search(r"\bwindow\s+(-?\d+)\b", match[6])
        count = int(explicit[1]) if explicit else step - previous
        if count <= 0 or count != step - previous:
            errors.append(f"window count mismatch at step {step}: {count}")
        if previous < warmup < step:
            errors.append(f"window at step {step} straddles warmup {warmup}")
        windows.append(dict(step=step, steps=count,
                            tok_s=rate if math.isfinite(rate) and rate > 0 else None,
                            loss=loss if math.isfinite(loss) else None))
        previous = step
    expected = list(range(log_every, steps + 1, log_every))
    if not expected or [row["step"] for row in windows] != expected or expected[-1] != steps:
        errors.append(f"missing/out-of-order step windows: expected {expected}, got {[w['step'] for w in windows]}")
    val_loss = None
    if len(final_values) != 1:
        errors.append("missing or duplicate FINAL VAL loss")
    else:
        try:
            value = float(final_values[0])
            if math.isfinite(value):
                val_loss = value
        except ValueError:
            pass
        if val_loss is None:
            errors.append("non-finite FINAL VAL loss")
    steady = [row for row in windows if row["step"] - row["steps"] >= warmup]
    if require_steady and not steady:
        errors.append("missing post-warmup windows")
    steady_steps = sum(row["steps"] for row in steady)
    rate = None
    if not errors and steady:
        rate = steady_steps / sum(row["steps"] / row["tok_s"] for row in steady)
    return dict(valid=not errors, errors=errors, windows=windows, steady_steps=steady_steps,
                steady_tok_s=rate, val_loss=val_loss)


def harmonic(rates):
    return len(rates) / sum(1.0 / value for value in rates)


def summarize(runs, planned_runs, profile):
    valid = len(runs) == planned_runs and all(run["valid"] for run in runs)
    summary = dict(valid=valid)
    if not valid or profile:
        return summary
    rates = {arm: [run["steady_tok_s"] for run in runs if run["arm"] == arm]
             for arm in ("baseline", "candidate")}
    averages = {arm: harmonic(values) for arm, values in rates.items()}
    pairs = []
    for start in range(0, len(runs), 4):
        group = runs[start:start+4]
        pairs.append(harmonic([group[1]["steady_tok_s"], group[2]["steady_tok_s"]]) /
                     harmonic([group[0]["steady_tok_s"], group[3]["steady_tok_s"]]))
    summary.update(baseline_tok_s=averages["baseline"], candidate_tok_s=averages["candidate"],
                   speedup=averages["candidate"] / averages["baseline"], pair_speedups=pairs,
                   run_ranges={arm: [min(values),max(values)] for arm, values in rates.items()},
                   val_losses={arm: [run["val_loss"] for run in runs if run["arm"] == arm] for arm in rates})
    return summary


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline_exe", type=Path)
    parser.add_argument("candidate_exe", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/tinystories")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--pairs", type=int, default=2, help="ABBA blocks (four runs each)")
    parser.add_argument("--warmup", type=int, default=20, help="excluded throughput steps, not LR warmup")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--profile", action="store_true", help="one run per arm, no throughput comparison")
    args = parser.parse_args(argv)
    try:
        if args.steps <= 0 or args.pairs <= 0 or args.log_every <= 0 or args.warmup < 0:
            raise ValueError("steps, pairs, log-every must be positive; warmup must be non-negative")
        if args.steps % args.log_every or args.warmup % args.log_every:
            raise ValueError("steps and warmup must align with log-every")
        if not args.profile and args.steps <= args.warmup:
            raise ValueError("benchmark needs post-warmup steps")
        binaries = {arm: path.resolve() for arm, path in
                    (("baseline",args.baseline_exe),("candidate",args.candidate_exe))}
        data_dir = args.data_dir.resolve()
        validation = data_dir / ("val.bin" if (data_dir / "val.bin").exists() else "validation.bin")
        dataset = dict(train=data_dir / "train.bin", validation=validation)
        for path in [*binaries.values(), *dataset.values()]:
            if not path.is_file():
                raise ValueError(f"required file is missing: {path}")
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        schedule = ["baseline","candidate"] if args.profile else ["baseline","candidate","candidate","baseline"] * args.pairs
        report = dict(mode="profile" if args.profile else "benchmark", valid=False,
                      created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      cwd=str(ROOT), steps=args.steps, warmup=args.warmup, log_every=args.log_every,
                      dataset={name: dict(path=str(path), size_bytes=path.stat().st_size,
                                          mtime_ns=path.stat().st_mtime_ns) for name,path in dataset.items()},
                      order=schedule, executables={arm: dict(path=str(path), sha256=sha256(path)) for arm,path in binaries.items()},
                      runs=[], summary=dict(valid=False))
        report_path = output / "report.json"
        write_report(report_path, report)
        for index, arm in enumerate(schedule, 1):
            run_dir = output / f"{index:02d}-{arm}"
            run_dir.mkdir()
            train_dir = run_dir / "train"
            train_dir.mkdir()
            env = build_environment(os.environ, data_dir=args.data_dir, run_dir=train_dir,
                                    steps=args.steps, log_every=args.log_every, profile=args.profile)
            settings = {key: value for key,value in env.items() if key.upper() not in SYSTEM_ENV}
            exe, log = binaries[arm], run_dir / "raw.log"
            digest = sha256(exe)
            started = time.monotonic()
            launch_error, exit_code = None, None
            with log.open("wb") as stream:
                try:
                    result = subprocess.run([str(exe)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    exit_code = result.returncode
                except OSError as error:
                    launch_error = str(error)
                    stream.write(("LAUNCH ERROR: " + launch_error + "\n").encode("utf-8"))
            elapsed = time.monotonic() - started
            parsed = parse_log(log.read_text(encoding="utf-8", errors="replace"), exit_code,
                               steps=args.steps, warmup=args.warmup, log_every=args.log_every,
                               require_steady=not args.profile)
            try:
                after = sha256(exe)
            except OSError:
                after = None
            if digest != report["executables"][arm]["sha256"] or after != digest:
                parsed.update(valid=False, steady_tok_s=None)
                parsed["errors"].append("executable changed during benchmark")
            run = dict(index=index, arm=arm, exe=str(exe), sha256=digest, sha256_after=after,
                       command=[str(exe)], settings=settings, exit=exit_code, elapsed_seconds=elapsed,
                       launch_error=launch_error, log=str(log), log_sha256=sha256(log), **parsed)
            report["runs"].append(run)
            report["summary"] = summarize(report["runs"], len(schedule), args.profile)
            report["valid"] = report["summary"]["valid"]
            write_report(report_path, report)
            print(json.dumps(dict(index=index, arm=arm, valid=run["valid"], exit=exit_code,
                                  steady_tok_s=run["steady_tok_s"], errors=run["errors"])), flush=True)
            if not run["valid"]:
                break
        print(json.dumps(dict(report=str(report_path), **report["summary"])), flush=True)
        return 0 if report["valid"] else 1
    except (OSError, ValueError) as error:
        print(f"benchmark configuration/error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
