#!/usr/bin/env bash
# Capture the method's regression + benchmark logs into this folder so the numbers in
# README.md / docs/ are backed by inspectable output. Run on the target device (RX 580).
#
#   MOTIFCL_BUILD=../build/dev ./capture_logs.sh [--data-path /path/to/tinyshakespeare.txt]
#
# Nothing here fabricates results: every line just runs a real binary/script and tees its
# output. Missing binaries are reported and skipped, not faked.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
build="${MOTIFCL_BUILD:-$repo/../build/dev}"
data_path=""
[ "${1:-}" = "--data-path" ] && data_path="$2"

run() { # name, command...
  local name="$1"; shift
  local log="$here/$name.log"
  echo "==> $name : $*"
  if "$@" >"$log" 2>&1; then
    echo "    ok  -> $(basename "$log")"
  else
    echo "    FAILED (exit $?) -> see $(basename "$log")"
  fi
}

echo "build dir: $build"
[ -d "$build" ] || echo "WARNING: build dir not found; ctest steps will fail until you build (see ../apply_to_motifcl.md)"

# device banner (best effort)
if command -v clinfo >/dev/null 2>&1; then
  run device clinfo
fi

# correctness regressions, including the memory truth gate
run ctest_method ctest --test-dir "$build" \
  -R "counter_state|counter_memory_truth|reversible_attn|f16_matmul_autograd" \
  --output-on-failure

# quality / isolation (PyTorch CPU is fine)
if command -v python >/dev/null 2>&1; then
  py=python
elif command -v python3 >/dev/null 2>&1; then
  py=python3
else
  py=""
  echo "WARNING: no python found; skipping PyTorch gates"
fi
if [ -n "$py" ]; then
  pg_args=()
  [ -n "$data_path" ] && pg_args=(--data-path "$data_path")
  run parity_gate "$py" "$repo/benchmarks/python/parity_gate.py" "${pg_args[@]}"
  run mlp_regression "$py" "$repo/benchmarks/python/mlp_regression.py"
  run counter_state_C_ablation "$py" "$repo/benchmarks/python/counter_state_C_ablation.py"
fi

echo "done. Commit the *.log files (and note which GPU/driver produced them)."
