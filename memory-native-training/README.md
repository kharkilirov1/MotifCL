# Memory-Native Training

A training method that attacks **all four memory pools of neural-network training at once**,
implemented and verified on a consumer GPU (AMD Radeon RX 580, OpenCL) on top of the
[MotifCL](https://github.com/) C++/OpenCL engine.

This repository **isolates the method** (sources, integration patches, regressions,
benchmarks, formal spec) as a self-contained research project. It is an **extension of
MotifCL**, not a standalone engine — the C++ sources use MotifCL's `Tensor` / `Backend` /
autograd / matmul. See [`apply_to_motifcl.md`](apply_to_motifcl.md) to build it.

> **Want it without the engine?** [`pytorch/`](pytorch/) is an **engine-independent PyTorch
> port** — `pip install` and run on stock CPU/CUDA, no MotifCL build. It realizes the learning
> dynamics and the optimizer-state saving today; the sub-byte packed-kernel win is on its
> roadmap (Triton). This is the path to making the method accessible to everyone, not just the
> OpenCL/RX 580 setup. Start there if you just want to use or reproduce the method.

## The method (two independent memory levers)

| Pool | Lever | On attention | Result (verified) |
|---|---|---|---|
| Parameters + optimizer + gradients | **Finite-state counter synapse** (0.75 byte/weight) | ✅ 1.28× parity | 0.75 byte/weight packed → **~16×** less than FP32+Adam; backward is memory-native (no dense grad_w, ≤1 transient weight) |
| Activations | **Reversible recompute** (forward stores nothing) | ✅ grad 2.8e-7 | naive-float recovery (3e-3) training-neutral at tested depth (≤12 blocks, Δce 0%) |

Plus enabling infrastructure: **f16 matmul autograd** (gradient match vs f32 ≈ 4e-4).

### Persistent state vs training peak (read this before quoting "0.75 byte/weight")
The two are different claims and we keep them separate:
- **Persistent state: 0.75 byte/weight** — measured, packed 6-bit, optimizer lives inside it.
- **Training peak: now also compact** — the backward never materializes a dense `[out,in]`
  weight-gradient (it is formed in registers inside the OpenCL kernel) and never carries a
  decoded dense weight across forward→backward (≤1 transient `[out,in]` alive at a time).
  This invariant is enforced by `tests/test_counter_memory_truth.cpp`. The compute price of
  the in-kernel gradient (~2× grad_w GEMM) means the **0.97× dense speed** number — measured
  on the earlier materialized path — must be re-measured on the fused path on the RX 580.

### Key empirical findings (real runs on RX 580, micro/tiny scale)
- Counter is **parity-class with dense FP32** on language tasks (char-LM 1.02×, attention-GPT 1.28×).
- Counter+row-RMS **beats ternary-QAT** (isolation −5.1% at d=256); the wall on exact FP32
  regression is **ternarization itself** (62× even for ternary-QAT+Adam), not the optimizer.
- Reversible recompute-backward is **correct through attention** (grad rel-err 2.8e-7) with
  **zero forward activation storage**.
- Scale tested is small (micro/tiny configs, ≤800 steps, one consumer GPU); 1B/7B figures in
  the docs are **memory arithmetic**, not training runs.

## Layout

```
engine/      our contribution to MotifCL
  kernels/   compact_counter.cl        (decode / row-stats / apply, packed 6-bit;
                                        *_fused_* variants form grad_w in-kernel)
  include/   compact_counter.hpp        (nn::CounterStateLinear)
  src/       compact_counter.cpp        (forward + memory-native fused backward node)
  patches/   matmul_f16_autograd.patch  (F16MatMulBackward)
             backend_kernel_route.patch (counter -> compact_counter.cl)
tests/       CTest regressions (counter teacher-recovery, memory truth gate,
             reversible-attention, f16 autograd)
results/     where to drop raw RX 580 logs for reproducibility (see results/README.md)
benchmarks/
  python/    PyTorch prototypes + parity/isolation gates (counter_state_*, parity_gate, mlp_regression)
  gpu/       standalone MotifCL witnesses (teacher-recovery, deep-MLP, char-LM, attention-GPT, perf/profile, reversible)
docs/        MEMORY_NATIVE_TRAINING_METHOD.md (formal spec) + COUNTER_STATE_NATIVE_DESIGN.md
```

## Reproduce
The method needs the MotifCL engine. Integration + build instructions are in
[`apply_to_motifcl.md`](apply_to_motifcl.md). Formal description: [`docs/MEMORY_NATIVE_TRAINING_METHOD.md`](docs/MEMORY_NATIVE_TRAINING_METHOD.md).

Python prototypes run standalone (PyTorch CPU):
```
cd benchmarks/python && python parity_gate.py
```
