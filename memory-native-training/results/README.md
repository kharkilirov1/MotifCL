# results/ — raw logs for reproducibility

This directory is where the **actual run logs** live so that the numbers quoted in
`README.md` and `docs/` are backed by something a reader can inspect, not just asserted.

> Honesty note: the logs are **not committed pre-filled**. They must be captured on the
> target device (the method is verified on an **AMD Radeon RX 580 / Polaris via OpenCL**).
> Whoever has that hardware runs `capture_logs.sh` (or the manual commands below) and
> commits the resulting `*.log` files. Until then, treat every performance/quality number
> in the docs as "claimed, pending the log in this folder".

## What to capture

| Log file | Command | Backs which claim |
|---|---|---|
| `ctest_method.log` | the `ctest -R ...` line below | correctness regressions incl. memory truth gate |
| `parity_gate.log` | `python parity_gate.py ...` | counter+RMS vs ternary-QAT / FP32 isolation |
| `mlp_regression.log` | `python mlp_regression.py` | counter+RMS vs ternary-QAT on FP32 regression |
| `counter_state_C_ablation.log` | `python counter_state_C_ablation.py` | C=11 (63 states) ablation |
| `counter_charlm_gpu.log` | `benchmarks/gpu/test_counter_charlm_gpu` | char-LM parity (1.02× dense) |
| `counter_gpt_gpu.log` | `benchmarks/gpu/test_counter_gpt_gpu` | attention-GPT parity (1.28×) |
| `counter_perf_gpu.log` | `benchmarks/gpu/test_counter_perf_gpu` | speed (re-measure fused path!) |
| `reversible_attn_gpu.log` | `benchmarks/gpu/test_reversible_attn_gpu` | recompute-backward grad 2.8e-7 |
| `device.log` | `clinfo` (or the engine's device banner) | which GPU/driver produced the above |

## Capture (Linux/macOS bash)

```bash
# from the MotifCL build that has the method integrated (see ../apply_to_motifcl.md)
MOTIFCL_BUILD=../build/dev ./capture_logs.sh
```

## Capture (manual)

```bash
# correctness, incl. the memory truth gate
ctest --test-dir "$MOTIFCL_BUILD" \
  -R "counter_state|counter_memory_truth|reversible_attn|f16_matmul_autograd" \
  --output-on-failure 2>&1 | tee results/ctest_method.log

# quality / isolation (PyTorch, CPU is fine). --data-path uses the real corpus;
# without it the gate falls back to a synthetic corpus and says so in the header.
python benchmarks/python/parity_gate.py --data-path /path/to/tinyshakespeare.txt \
  2>&1 | tee results/parity_gate.log
```

When you add logs, also note in the commit **which GPU + driver** produced them so the
numbers are attributable.
