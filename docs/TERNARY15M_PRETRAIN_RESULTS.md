# Ternary pretrain from scratch on an RX 580 — run 1 results

**2026-07-24 20:49 → 2026-07-26 20:53, 48 h wall, one Radeon RX 580 8 GB (Polaris, 2017).**

A stories15M-shaped language model trained **from random init** with every
attention/FFN linear as a `CounterStateLinear` ternary counter synapse. The
forward pass never saw anything but ternary weights, and no FP32 latent copy of
those weights existed at any step — the counter state IS the model.

Reference for comparison: [brianbell-x/ternary15M](https://github.com/brianbell-x/ternary15M),
the same architecture trained BitNet-b1.58 style (STE over FP32 latents) on one
L40S for ~50 min / $0.70.

## Headline numbers

| | val loss | ppl |
|---|---:|---:|
| first eval (step 500) | 3.6017 | 36.7 |
| **best (step 155 000)** | **1.8018** | **6.06** |
| final (step 160 000) | 1.8209 | 6.18 |
| reference, hard ternary | 1.6074 | 4.99 |
| reference, latent STE eval | 1.5970 | 4.94 |

Run health: 160 000 steps, 655 M tokens, **4152 tok/s average** over 1600 log
lines, zero NaNs, zero restarts. Only 11 of 1600 log lines fell below 3500 tok/s
(a period when another process shared the GPU).

## Configuration

Architecture (identical to stories15M): dim 288, 6 layers, 6 heads (MHA,
kv=6), SwiGLU hidden 768, vocab 32 000, seq 256, RoPE theta 1e4, RMSNorm
eps 1e-5.

Ternary body: 36 `CounterStateLinear` modules (6 per block — wq/wk/wv/wo plus a
fused gate+up and down; the reference counts 42 because it keeps gate and up
separate), C=11, rms_beta 0.9, rms_eps 1e-3, lr_scale 0, init_gain 1.0.

Training: BATCH=16 x SEQ=256 = **4096 tokens/step**, no gradient accumulation.
Counter lr cosine 4e-3 -> 4e-4 (min 10 %), warmup 2 % = 3200 steps. FP params
(embedding, untied head, norms) on AdamW, cosine 6e-3 -> 6e-4, betas
(0.9, 0.95), eps 1e-8, wd 0.1 on embedding+head only. Eval every 500 steps on a
fixed 96-record slice; checkpoint every 1000 steps; seed 1337.

Data: TinyStories tokenized by the reference `preprocess.py` (llama2 32k SP
tokenizer) — 2 009 354 train records, 20 210 val records of seq+1 uint16.

## Memory: what the method actually buys

Checkpoint 216 MB total:

| component | size |
|---|---:|
| **ternary body — counter state of all 36 layers (5.97 M weights)** | **4.4 MB** |
| counter per-row scale + RMS moment | 0.3 MB |
| RMSNorm gains | 0.05 MB |
| FP32 embedding + untied head | 71 MB |
| AdamW moments for those two matrices | 141 MB |

The trainable fabric of the model is **4.4 MB, 2 % of the checkpoint**. During
training the counter body carried no optimizer moments and no dense weight
gradients at all. Peak VRAM was 3.9 GB (sampled 3910 / 2644 / 3622 MB across a
step) and went almost entirely to activations and the 32k-vocab logits chain —
4096 tokens x 32000 vocab x 4 B = 524 MB per tensor, three of them live
(logits, softmax, grad).

This mirrors the reference author's own conclusion: make the obvious thing
cheap and the real bottleneck shows itself. For them the FP32 embedding became
70 % of the model; for us the vocabulary head and its Adam state became 98 % of
the checkpoint.

## Shape of the curve

1. **Collapse, steps 500-3000**: 3.60 -> 2.30. Token statistics learned.
2. **Log-linear descent, 3000-11000**: 2.30 -> 2.00.
3. **Noise-ball plateau, 11000-90000**: 2.00 -> 1.87 over 79 000 steps. Points
   oscillate in a +/-0.01 corridor; five separate downward "staircases" of 3-7
   consecutive points appeared and each dissolved back into the corridor.
   At fixed lr the stochastic-rounding counter updates keep flipping synapses
   even where the mean gradient is zero — lr acts as a temperature, and the
   smallest expressible weight change is a whole flip, so there is no
   fine-adjustment mode.
4. **Cooldown, 90000-160000**: 1.87 -> 1.80. The descent rate roughly tripled
   (-0.0014 per 1000 steps vs -0.0005 on the plateau) as lr fell from 2e-3 to
   4e-4. Real, steady, but no cliff-edge drop.

## Why we land 0.20 above the reference

Two measurable causes, both addressable:

1. **Effective batch is 16x smaller.** The reference uses
   `--gradient-accumulation 16`: 65 536 tokens per optimizer step against our
   4096. Gradient noise scales as 1/sqrt(B), so our per-update noise is ~4x
   higher — and for a discrete accumulator that noise converts directly into
   spurious flips. Plain accumulation is not available to us: counter layers
   self-update **inside** backward, so one backward equals one state update.
   Raising it needs either a larger micro-batch (memory-bound) or an
   accumulate-then-update mode in `CounterStateLinear`.
2. **The accumulator itself is bounded.** BitNet-style STE keeps a continuous
   FP32 latent: 24 bits of mantissa, unbounded evidence accumulation, ternary
   only in the forward. Our counter is a uint8 automaton at C=11 — about 6 bits
   of resolution with hard saturation. That is the structural price of not
   storing latents, and the most plausible reason the floor sits higher.

Minor deviations from the reference, listed for completeness: untied LM head
(the Vulkan `matmul_transpose_b` path records no autograd node, so a tied head
silently severs the backward chain), no gradient clipping on the FP params
(reference uses 1.0), fused gate+up.

## What run 2 should change, in order of expected effect

1. Chunked vocabulary loss + BATCH=32 — attacks the dominant noise source and
   frees over a gigabyte of VRAM at the same time.
2. Tied LM head — removes 37 MB of weights and 74 MB of Adam moments; needs the
   `matmul_transpose_b` autograd node on Vulkan first.
3. Gradient clipping at 1.0 on the FP params, matching the reference.
4. As a separate experiment: C=15 or C=21 against accumulator saturation. This
   tests the structural hypothesis directly — if the floor drops, the bound on
   the counter is what limits quality; if it does not, batch noise was the whole
   story.

Reversible blocks are **not** the lever here: they compress layer activations,
but our peak is dominated by the vocabulary logits chain, which lives outside
the blocks. That was an early misread on my part, corrected by measurement.

## Reproduce

```bash
cmake --preset release
cmake --build build/release --target motifcl_train_ternary15m -j 8
DATA_DIR=data/tinystories OUT_DIR=out/ternary15m_run1 \
STEPS=160000 BATCH=16 EVAL_EVERY=500 CKPT_EVERY=1000 LOG_EVERY=100 \
  ./build/release/tools/motifcl_train_ternary15m
```

Data preparation uses the reference repo's `scripts/preprocess.py` with the
llama2 `tokenizer.model`; the runner reads `train.bin` + `val.bin`
(or `validation.bin`) of packed uint16 records.
