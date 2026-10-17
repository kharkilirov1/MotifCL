# MN-MoE15M — a sparsely-routed counter-synapse MoE for the RX 580 (design record)

Status: **design, not a result.** 2026-10-16. Base revision: `09cadbe` (working tree has unrelated
untracked FOG NL work; nothing here depends on it).

Evidence policy: every claim below names the repository, path and line it came from. No training
number in this file is measured — the only measured baselines quoted are the previously recorded
`ternary15m` run-1 numbers, cited with their source.

## 1. Question

Can a **sparsely-routed Mixture-of-Experts** language model be designed, implemented and trained
**from random init on a single Radeon RX 580 8 GB** through MotifCL's native Vulkan FP32 path —
i.e. without ROCm, CUDA, tensor cores, or a PyTorch dependency?

## 2. What the runtime already provides (verified, not assumed)

| Capability | Evidence | Status |
|---|---|---|
| Hard top-1 router over 7 candidates, Vulkan, **forward + logits backward + candidate backward** | `kernels/vulkan/fog_hard_route7_f32.comp`, `..._logits_bwd_f32.comp`, `..._candidate_bwd_f32.comp`; op `src/ops/fog.cpp:250-278`; node `src/ops/fog.cpp:122-160` | ready |
| MoE expert-gradient semantics already correct | `..._candidate_bwd_f32.comp:12` zeroes every row where the candidate did not win | ready |
| Straight-through softmax router gradient | `..._logits_bwd_f32.comp:1-2,31-37` — `grad_logits = p*(gp - Σ p·gp)` | ready |
| Counter-synapse linear with in-backward fused state update | `include/motifcl/nn/compact_counter.hpp:16-47`, `src/nn/compact_counter.cpp:25-51` | ready |
| 2-D matmul fwd + composed backward on Vulkan | `src/ops/matmul.cpp:1390`, backward at `:1131`, `:1225` | ready |
| SwiGLU / SiLU fwd + bwd, RMSNorm fwd + bwd, CE fwd + bwd, Adam/SGD | kernels in `kernels/vulkan/`; dispatch in `src/runtime/vulkan_backend.cpp` | ready |
| Per-row bias add on Vulkan (router bias) | `kernels/vulkan/add_bias_rows_f32_rb4.comp` | ready |
| Proven from-scratch training loop on this device | `tools/train_ternary15m.cpp`; recorded result `docs/TERNARY15M_PRETRAIN_RESULTS.md` (288 dim / 6 layers / 32k vocab, 160k steps, 655M tokens, **4152 tok/s**, peak **3.9 GB**, val ppl 6.06) | measured |
| Data already prepared | `data/tinystories/train.bin` (1.03 GB = 2 009 354 records × 257 uint16), `validation.bin` | present |

Not needed for this design, and consequently **not** a blocker: GPU top-k/sort kernels, grouped
GEMM, generic gather/scatter, capacity factor, aux-loss op. (For the record they do not exist:
no top-k/sort/argsort kernel in either backend, rank-2 matmul only, gather/scatter exist in
embedding form only — see `kernels/vulkan/` inventory.)

## 3. Design

### 3.1 Backbone — unchanged from the proven configuration

dim 288, 6 layers, MHA-6 (kv=6), RoPE θ=1e4, RMSNorm eps 1e-5, vocab 32 000, seq 256,
**tied** LM head (the `matmul_transpose_b` autograd node now exists: `src/ops/matmul.cpp:1128-1139`),
FP32 trainables = token embedding + norms on AdamW/Adam.

### 3.2 MoE FFN block (replaces the dense SwiGLU FFN)

Per layer:

```
x ──► rmsnorm ──► router logits  = x @ W_r + b            W_r [288,7] f32, b [7] (balancing knob)
             └──► expert_e(x)  = W_down_e · swiglu(W_up_e · x)      e = 0..6   (counter synapses)
y  = fog_hard_route7(logits, {expert_0(x) … expert_6(x)})   # top-1 selection, straight-through grad
y += shared(x)                                              # always-on counter FFN (192 wide)
out = residual + y
```

- **7 experts** — fixed by the existing kernel (`fog_hard_route7` is hard-coded to 7 candidates).
- Expert width **192** (`W_up: 288→384`, SwiGLU, `W_down: 192→288`), shared expert width 192.
- Every expert linear is `nn::CounterStateLinear` (C=11, rms_beta 0.9, rms_eps 1e-3): **no FP32
  master weights and no optimizer moments for expert weights** — the 6-bit counter state is the expert.
- Router weights are the only new FP32 trainable (288×7 = 2 016 parameters).

### 3.3 Load balancing

Aux-loss-free, DeepSeek-v3 style: per-expert bias `b_e` added to the router logits before argmax,
never to the probabilities; after each step (outside the Vulkan batch) the token counts are read
back and `b_e += γ · sign(mean_load − load_e)`. This needs no new gradient path and no aux loss.
A Switch-style aux loss is the fallback if the bias scheme measures badly.

### 3.4 Compute budget (analytic, to be measured)

Per token, per layer, MACs: attention ≈ 4d² + 2dT = 0.48 M. FFN:

| variant | FFN MACs/token/layer | FFN hidden capacity | active |
|---|---:|---:|---:|
| dense baseline (`ternary15m`) | 3·288·768 = 0.66 M | 768 | 768 |
| MN-MoE15M (7 experts × 192 + shared 192, all computed) | 7·3·288·192 + 3·288·192 = 1.33 M | 1 536 | 384 |

Whole model per token: baseline ≈ 16.1 M MACs (attention 2.9 M + FFN 4.0 M + logits 9.2 M);
MN-MoE15M ≈ 18.7 M MACs → **≈ 1.16×** the baseline compute, i.e. expected **≈ 3 400–3 900 tok/s**
against the measured 4 152 tok/s, while *capacity* grows 2× with 1/7 activation sparsity.

Note the honest framing: the **routing is sparse, the compute is not** — `fog_hard_route7` selects
rows from 7 fully-computed candidates. True sparse dispatch (compute only the winners) needs
gather/scatter wrappers and is a v2 item, not a v1 claim.

### 3.5 Memory (estimate, to be measured at step 0)

Baseline peak 3.9 GB was dominated by the 4096×32000 FP32 logits chain (3 × 524 MB). MoE adds
7 expert outputs [4096,288] (33 MB per layer, live for the route kernel) plus their autograd
graphs. Expect ~4.5–6 GB. Mitigations already known: `BATCH 8` instead of 16, and/or chunked
vocabulary loss (the top recommendation recorded in `docs/TERNARY15M_PRETRAIN_RESULTS.md:135-137`).

## 4. Known risks and their disposition

1. **Expert count fixed at 7.** A general top-k needs new SPIR-V; `glslc` is not installed on this
   host (no Vulkan SDK), and `kernels/vulkan/*.comp` cannot be regenerated without it. v1 therefore
   uses E=7. Unblocking v2 = install LunarG SDK (or a standalone shaderc build) once.
2. **Zero-gradient experts still tick.** Non-winning experts receive an all-zero `grad_out`; the
   fused in-backward update then leaves codes untouched but decays the row RMS `v` and advances the
   RNG seed. This is expected and must be *measured*, not assumed harmless.
3. **Host reads cannot happen inside an open Vulkan batch** (`examples/cpp/22_fog_v3_nl_binder_train.cpp:104-105`).
   Router logits may be read back for balancing only after `runtime.batch_end()`.
4. **No gradient accumulation** (one backward = one counter update), inherited from the baseline.
5. **Tied head costs ~8 % throughput** in the recorded probe (`TERNARY15M_PRETRAIN_RESULTS.md:112-132`);
   accepted for the halved checkpoint and reference parity.

## 5. Implementation plan and gates

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | `include/motifcl/nn/moe7.hpp` + `src/nn/moe7.cpp`: 7-expert block wiring `fog_hard_route7` to `CounterStateLinear` experts, shared expert, router parameter + bias | **done** — `tests/test_moe7_reference.cpp` (registered in ctest): forward matches a host reference at rel_err 1.17e-07, module token counts equal the host argmax distribution (mismatch 0), router gradient non-zero, exactly the routed experts receive gradient (`leaked=0`) |
| P1 | `tools/train_moe15m.cpp` (clone of `train_ternary15m.cpp`) with `E`, `HIDDEN_EXP`, `HIDDEN_SHARED`, `BALANCE_GAMMA`, `BATCH`, `SEQ` knobs | **done** — 5-step smoke finite and deterministic; knobs are `HIDDEN_EXP`, `HIDDEN_SHARED`, `BALANCE_GAMMA`, `AUX_WEIGHT`, `SAVE_CKPT`; checkpoint/resume verified (step 300 -> 305) |
| P2 | Step-0 memory/throughput gate: batch and width grid, recorded into `reports/` | **done, superseded** — B8xT256 is the working point: 2010-2062 tok/s and 3359 MB device peak with `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1`; B16xT256 hits `VK_ERROR_DEVICE_LOST`. The measured lever was memory placement, not batch/shape (see `MOE_PRETRAIN_RESULTS.md` §2-3) |
| P3 | Short pretrain (≈ 5–10 k steps) vs the recorded dense baseline at matched tokens | 300-step curve measured (val 4.8114/3.9558/3.7365 at 100/200/300, expert spread 0.028); a 20 000-step matched pair is in flight |
| P4 | Long run (multi-day, resumable), then eval + write-up | started at 20 000 steps (41 M tokens); the shared horizon was shortened to 12 000 steps (24.6 M tokens) at the step-2000 checkpoint so the MoE and both dense controls share one schedule; checkpoints every 2 000; resumable and resumed for real (step 2000 -> continuation) |

## 6. Training recipe (the exact configuration in use)

Everything below is a single env-var-driven command; there is no config file.

| element | value |
|---|---|
| corpus | TinyStories, packed `uint16`, record = seq+1 = 257 (`data/tinystories/train.bin` 2 009 354 records, `validation.bin` 20 210) |
| backbone | dim 288, 6 layers, MHA 6 heads / 6 KV heads, RoPE theta 1e4, RMSNorm eps 1e-5, vocab 32 000, seq 256, SwiGLU, **untied** head (`TIE_HEAD=0`) |
| MoE FFN | 7 experts x hidden 192 + always-on shared expert 192; `CounterStateLinear` C=11, `rms_beta` 0.9, `rms_eps` 1e-3, `init_gain` 1.0 |
| batch | `BATCH=8`, `SEQ=256` -> 2 048 tokens/step, **no gradient accumulation** (a counter layer updates inside each backward) |
| learning rates | counter peak `LR=4e-3`; FP peak `LR_FP=6e-3` for embedding, norms and the router weights (Adam, beta 0.9/0.95, eps 1e-8, weight decay 0.1 on the embedding only) |
| schedule | linear warmup over 2 % of steps, cosine decay to 10 % of the peak; horizon `STEPS=12000` (24.6M tokens) shared with the dense control after the step-2000 checkpoint (started at 20000) |
| load balancing | Switch auxiliary loss applied as an explicit router gradient, `AUX_WEIGHT=0.01`; aux-loss-free selection bias updated by `BALANCE_GAMMA=0.01` sign steps, clamped to +-2 |
| memory policy | `MOTIFCL_VK_PREFER_DEVICE_LOCAL=1` — **required** on this box: the current default (host-visible/GTT) costs 2.0x on the MoE and 2.8x on the dense model at bit-identical losses (`MOE_PRETRAIN_RESULTS.md` §2) |
| evaluation | 96 validation records every 500 steps with counters in inference mode (no state updates) |
| checkpointing | every 2 000 steps plus `meta.txt`; `RESUME=1` restores counters, router, bias, Adam moments and the RNG seeds |
| measured cost | 2 010-2 062 tok/s, ~1.0 s/step, device peak 3 359 MB, working set ~1.3 GB (activations live in VRAM) |

## 7. What is deliberately *not* claimed

- No sparse-execution speedup: v1 computes all 7 experts.
- No quality prediction from the parameter count; capacity is not quality.
- No VRAM, tok/s or convergence number before it is measured on this device.
- No comparison against the external L40S BitNet reference — that number is not reproducible here.

## 8. Next experiments, ranked by expected value

1. **Sparse dispatch with existing kernels (no new SPIR-V).** Feasibility was checked against
   the kernel sources before writing any code, and it splits into one solved problem and one
   structural constraint:
   - *Row gather/scatter exist and are general.* `embedding_gather_f32_i32` is
     `out[t,d] = weight[indices[t], d]` with an OOB index writing `0`, and
     `embedding_weight_backward_f32_i32_scatter{,_cas}` is `grad_weight[indices[t],d] +=
     grad_out[t,d]`. Passing the activations as `weight` with `vocab_size = rows` turns both into
     generic row primitives, and their composition gives the autograd pair (gather's backward is
     the scatter, the scatter's backward is the gather) plus free padding: a padded index of `-1`
     yields a zero row, and `SwiGLU(0) = 0`, so padded slots contribute exactly nothing.
   - *Atomics are not a blocker on this GPU.* The native scatter needs
     `VK_EXT_shader_atomic_float`, which `VulkanCaps::supports_atomic_float` reports **false** for
     GCN4/RX 580. `embedding_weight_backward_f32_i32_scatter_cas.comp` exists precisely for that
     case ("notably GCN4 (Radeon RX 580), where the native extension is exposed but collapses
     every scattered add onto a single address"; integer `atomicCompSwap` is core Vulkan 1.0),
     and `src/nn/embedding.cpp` already selects it when `!supports_atomic_float`.
   - *Structural constraint found:* the router decision needs the logits on the host, and a host
     read **inside an open batch** returns pre-submission (stale) data, because batch dispatches
     are deferred to `batch_end`. The guard in `VulkanBuffer::upload` only forbids *writes* into a
     buffer an open batch references, but the staleness alone rules the naive version out.
     Consequences: sparse routing forces either a GPU-side top-k (blocked on `glslc`) or
     **per-layer batching** (two submits per block, ~12 per step, each ending in a fence wait) in
     the trainer. The second is implementable without new kernels and is the cheap path to test.
   Expected payoff if it works: per-token FFN MACs fall from `7*3*d*h_e + 3*d*h_s` to
   `3*d*h_e + 3*d*h_s`, i.e. about 1.6x less total compute once the vocabulary head is counted —
   the only route to a MoE that is *faster* than the dense control rather than 2.5x slower.
   Not implemented yet; the design record keeps the measured v1 numbers as the current claim.
2. **General top-k / capacity / expert-choice routing.** Blocked on `glslc` (absent); a single
   `topk_rows_f32.comp` plus a grouped GEMM would remove the E=7 constraint.
3. **Capacity vs active width.** `moe_e384` (active 576) against `dense_h384` (active 384)
   isolates how much of any quality difference comes from capacity rather than from routing.
4. **Balancing budget.** `moe_aux3e3` / `moe_aux0` plus `BALANCE_CLAMP` sweep: the current
   spread ~0.03 may be over-regularised, and the clamp saturation observed at +-2 shows the
   bias stops adapting on long runs.
5. **Chunked vocabulary loss.** The 32k-vocab logits chain dominates both memory and FLOPs
   (three live `4096x32000` tensors is 524 MB each at B16); chunking it would cut the device
   peak roughly in half and is the largest single memory lever for larger batches.
6. **Second-tier routing (shared expert always on) is already in v1** — a top-2 variant would
   need items 1-2 first.
