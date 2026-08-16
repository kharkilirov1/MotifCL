# Review response — making the backward memory-native

This addresses the review ("формат состояния уже compact, но backward ещё не compact").
Each recommended item below is mapped to the concrete change. The central fix: the backward
no longer keeps a dense decoded weight per layer, and never materializes a dense weight-
gradient — grad_w is formed in registers inside the OpenCL kernel.

| # | Review item | Status | Where |
|---|---|---|---|
| 1 | Remove saved dense `w` from `CounterBackwardNode` | done | `engine/src/compact_counter.cpp` — node stores only `x`; `w` decoded as a scoped temporary in `backward()` just for `grad_x`, freed immediately. |
| 2 | Make the update node independent of `x.requires_grad()` | done | `forward()` attaches the node when `is_enabled() && update_enabled_`; `backward()` calls `x.backward(grad_x)` only `if (x.requires_grad())`. A counter layer fed raw input now self-updates. |
| 3 | `parity_gate.py --data-path` + offline fallback | done | `benchmarks/python/parity_gate.py` — explicit path → cached file → download → deterministic synthetic corpus; prints the source it used. |
| 4 | Save raw RX 580 logs in `results/` | scaffolded | `results/README.md` + `results/capture_logs.sh`. Logs are **not fabricated** — they must be captured on the RX 580 (see note below). |
| 5 | Replace full `grad_w` with row-tile `grad_w` | superseded by #6 | The row-tile step is subsumed: the fused path materializes **no** grad_w at all (not even a tile). |
| 6 | Replace row-tile `grad_w` with a true fused OpenCL update | done (needs on-device perf re-measure) | `engine/kernels/compact_counter.cl` — `counter_row_stats_fused_f32` / `counter_apply_update_fused_f32` compute `grad_w[o,i]=Σ_n grad_out[n,o]·x[n,i]` in registers; host path `apply_update_fused`. |
| 7 | Add a peak-memory truth gate | done | `tests/test_counter_memory_truth.cpp` — fails if peak live dense `[out,in]` weight > 1 or any dense grad_w is allocated. Backed by host counters in the layer. |
| 8 | Only then claim "0.75 byte/weight training peak"; reframe status | done | README + `docs/` split **persistent state (0.75 B/w, measured)** from **training peak (now compact, gate-enforced)**; softened the reversible "fixed-point not needed" to a tested-depth claim with a depth-sweep/anchor fallback. |

## What is verified vs what is not

- **Verified by reading / by construction**: the two leaks are gone in code; the truth gate
  encodes the invariant; the numerics of the fused kernel are an exact transcription of the
  materialized path (`grad_w[o,i]` is the same sum), so teacher-recovery should be unchanged.
- **Verified locally**: the Python `--data-path`/offline-fallback control flow (the C++ side
  cannot be built here — no OpenCL device in this environment).
- **NOT yet verified on hardware**: the fused C++/OpenCL path has **not been compiled or run
  on the RX 580 from this change**. It must be built and `ctest`-run on the target device
  (`apply_to_motifcl.md`), and the **0.97× dense speed must be re-measured** on the fused
  path — it costs ~2× the grad_w GEMM (grad_w is recomputed in row-stats and in apply).

## Honest caveat on the logs

`results/` is intentionally empty of data. Producing the RX 580 logs requires the hardware;
this change cannot fabricate them. Whoever has the Polaris box runs `results/capture_logs.sh`
and commits the `*.log` files alongside the GPU/driver banner.

## Suggested follow-up (not done here)

- Re-measure fused-path perf on RX 580; if the ~2× grad_w recompute hurts, cache the
  row-stats grad_w in work-group **local** memory (still not global) instead of recomputing
  it in apply.
- Optionally fuse `grad_x` too (compute `grad_x[n,i]=Σ_o grad_out[n,o]·s_o·t_oi` without
  decoding a dense `w`), which would drop the peak from ≤1 transient weight to 0.
- Reversible depth-sweep + anchor fallback before any LLM-depth claim.
