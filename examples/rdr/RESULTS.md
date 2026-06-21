# RDR mechanism study — training runs

Full RDR model trained on the bounded verifiable-task dataset (3 families,
difficulty == reasoning depth). This is a **mechanism probe at toy scale**, not a
capability result.

## Setup

| | |
|---|---|
| Model | `d_model=96`, `n_recur=6`, full RDR (recurrence + AttnRes + RLVR) |
| Params | **755.8K** |
| Dataset | `build_dataset.py` defaults — train 3801 / val_interp 647 / test_extrap 2400 |
| Train | 3000 steps, batch 48, Adam lr 2e-3, exact-verifier RLVR |
| Eval | **real autoregressive generation** scored by exact `verify()`, n=300/split |
| Wall time | ~24 min, CPU (4 cores) |

Two runs are recorded:

- **Run 1** — original trainer. (`rdr_results_full.json`, `train_full.log`)
- **Run 2** — after fixing two depth-credit bugs (`rdr_results_full2.json`,
  `train_full2.log`). Folds the depth cost into the REINFORCE return (the old
  `ponder*depth` term had **zero gradient**, since depth is sampled/detached) and
  centres the advantage with a **per-difficulty baseline**.

## Headline numbers

| | val_interp acc | test_extrap acc | depth trend by difficulty (eval) |
|---|---:|---:|---|
| **Run 1** | 0.223 | 0.147 | flat ~3.2–3.6 |
| **Run 2** (bugs fixed) | **0.237** | **0.150** | flat ~3.1–3.7 (slightly **decreasing** on the hard tail) |

```
Run 2  val_interp  acc 0.237 | family modchain .15 ptrchase .26 eval .33
                              | diff d1 .22 d2 .31 d3 .23 d4 .20
                              | depth d1 3.67 d2 3.06 d3 3.49 d4 3.52
Run 2  test_extrap acc 0.150 | family modchain .09 ptrchase .29 eval .05
                              | diff d5 .12 d6 .18
                              | depth d5 3.35 d6 3.31
```

## What the bug fix changed (and what it didn't)

**It changed the training dynamics.** With the depth cost actually reaching the
router, mean depth became *dynamic* — over Run 2 it swung 2.8 ↔ 5.1 as the router
traded depth against reward, instead of drifting near-constant. So the
credit-assignment machinery now works: the router *can* and *does* modulate depth
in response to the exact-verifier reward.

**It did not produce difficulty-proportional depth.** At eval (deterministic
argmax depth) the per-difficulty trend is still **flat, and if anything decreasing**
on the extrapolation difficulties (d5/d6 ≈ 3.3 < d1 ≈ 3.7). End-task accuracy moved
only within noise (+0.014 interp, +0.003 extrap).

## Conclusion (now a robust null)

After removing the two mechanisms that could have masked the signal, the central
hypothesis — *the router allocates more recursion depth to harder problems* — is
**still not supported** at this scale. This is a stronger negative result than Run 1
because we can no longer attribute it to broken credit assignment.

The most likely reason is **mechanistic, not a tuning issue**: in this dataset
"difficulty" = number of task *hops*, but the architecture is never forced to
unroll one recurrence step per hop. The answer is a short token span, the whole
input (which already contains `k=…`) is processed in parallel each forward pass,
and the shared recurrent block can learn a *parallel shortcut* that solves d1 and
d6 with the same depth. Depth and task-hops are simply not tied, so the router has
no pressure to scale one with the other.

### What would actually test "harder ⇒ deeper" (next experiments)

- **Force sequential unrolling**: a task whose answer literally cannot be produced
  without N sequential steps (e.g. emit the *full trajectory* of the modchain /
  pointer-chase, not just the final value) — then depth is on the critical path.
- **Couple depth to the loss explicitly**: supervise/curriculum the router so
  early training only rewards correctness when depth ≥ hops, then relax.
- **Drop teacher forcing for the reward**: score depth against *autoregressive*
  rollouts so the reward reflects the harder, depth-sensitive regime.
- **Ablation grid** (`train.py --ablate`) on the forced-unrolling task, to check
  whether recurrence / AttnRes / RLVR each contribute once depth is load-bearing.

## Training curve (Run 2, teacher-forced exact-match reward)

```
step    1 | ce 5.68 | reward 0.00 | depth 3.5
step  600 | ce 0.93 | reward 0.33 | depth 4.8
step 1000 | ce 1.05 | reward 0.19 | depth 5.1
step 1700 | ce 0.80 | reward 0.33 | depth 2.8
step 2400 | ce 0.59 | reward 0.62 | depth 3.3
step 3000 | ce 0.58 | reward 0.60 | depth 3.3
```

---

# Experiment 3 — forced-unrolling + ablation grid

To put depth on the critical path, `build_traj.py` changes the target from the
**final value** to the **full trajectory** of intermediate states (one per hop):
emitting state *t* requires state *t-1* plus exactly one more step, so the
iteration cannot be solved by a single parallel shortcut. Difficulty == hops ==
trajectory length == sequential steps. Then the section-6 ablation grid is run on
this task (4 configs × 2000 steps, batch 48, `--max_new 28`, eval n=150/split).

## Ablation results (exact full-trajectory match)

| Config | `use_recurrence` | `use_attnres` | `use_rlvr` | **interp** | **extrap** |
|--------|:---:|:---:|:---:|---:|---:|
| full RDR        | ✓ | ✓ | ✓ | 0.080 | **0.000** |
| − RLVR          | ✓ | ✓ | ✗ | 0.087 | 0.000 |
| − AttnRes       | ✓ | ✗ | ✓ | 0.087 | 0.000 |
| − recurrence    | ✗ | ✗ | ✓ | **0.120** | 0.000 |

Mean router depth per difficulty (interp), still **flat** in every recurrent config:

```
full RDR    d1 3.15  d2 3.39  d3 3.25  d4 3.27
- RLVR      d1 3.80  d2 3.51  d3 3.73  d4 3.78
- AttnRes   d1 2.81  d2 2.58  d3 2.67  d4 2.73
- recurrence  (depth fixed at 1.0)
```

Per-difficulty accuracy falls off a cliff after 2 hops in *every* config
(e.g. full RDR d2 0.21 → d3 0.03), i.e. the model learns a shallow ≤2-step
pattern rather than the iteration.

## What this tells us

1. **Length extrapolation is the real wall: every config scores 0.000 on the
   harder split.** Forcing the trajectory output makes systematic generalisation
   *strictly harder* than the final-value task (which managed ~0.15) — errors
   compound over the longer required output and nothing here generalises the
   iteration to unseen lengths. This is the clearest single finding.
2. **The recurrence machinery does not earn its keep at this scale — it slightly
   hurts.** Removing components never lowers interp accuracy, and the *simplest*
   model (no loop, depth 1) is the **best** (0.120 vs 0.080 for full RDR). The
   loop adds optimisation difficulty and RL variance without a capability gain on
   these tasks.
3. **Depth is still flat across difficulty**, now confirmed under a task where
   sequential computation is unavoidable — because the unrolling lives on the
   *token* axis (one emitted state per hop), not the *depth* axis, so per-token
   depth need not grow with difficulty either.

### Bottom line for the mechanism study

Across three experiments — final-value, bug-fixed credit assignment, and
forced-unrolling — the central RDR claim (*adaptive depth that scales with
problem difficulty and improves systematic generalisation*) is **not supported at
toy scale on these tasks**. The honest reading is that this probe sharpens nothing
latent here: a depth-1 model is as good or better, and the binding constraint is
**length/compositional generalisation**, which none of recurrence, AttnRes, or
RLVR address. Testing the claim properly would need either much larger scale (the
open MoR / Kimi-Linear / Titans implementations the spec points to) or a task
where a *single* output token provably requires difficulty-many *sequential*
internal steps that cannot be offloaded to the token axis.

## Training curve (Run 2, teacher-forced exact-match reward)

```
step    1 | ce 5.68 | reward 0.00 | depth 3.5
step  600 | ce 0.93 | reward 0.33 | depth 4.8
step 1000 | ce 1.05 | reward 0.19 | depth 5.1
step 1700 | ce 0.80 | reward 0.33 | depth 2.8
step 2400 | ce 0.59 | reward 0.62 | depth 3.3
step 3000 | ce 0.58 | reward 0.60 | depth 3.3
```

Artifacts (`rdr_dataset.jsonl`, `rdr_traj.jsonl`, `rdr_results_full*.json`,
`rdr_traj_ablation.json`, `*.log`) are git-ignored; regenerate with the commands
in `README.md`.

---

# Experiment 4 — does the model behave differently at inference?

Reasonable worry: training uses **teacher forcing** and **sampled** router depth,
while inference is **autoregressive** with **argmax** depth — so maybe the reported
numbers misrepresent inference. `diagnose.py` measures both gaps directly on one
full RDR trained on the trajectory task (2500 steps), per difficulty.

```
=== val_interp (n=200) ===
diff |  TF acc  AR acc    gap | argmax d   sampled d (mean±std)
  1  |  0.143   0.143   0.000 |   3.39       3.39 ± 0.51
  2  |  0.185   0.185   0.000 |   3.75       3.75 ± 0.46
  3  |  0.044   0.044   0.000 |   3.61       3.61 ± 0.51
  4  |  0.061   0.061   0.000 |   3.58       3.59 ± 0.52
=== test_extrap (n=200) ===
  5  |  0.000   0.000   0.000 |   3.77       3.77 ± 0.58
  6  |  0.000   0.000   0.000 |   3.71       3.72 ± 0.50
```

**Both suspected gaps are ~zero, and that is not a coincidence:**

1. **Teacher-forced vs autoregressive accuracy is identical at every difficulty
   (gap 0.000).** With *greedy* decoding and an *exact-match* metric the two
   coincide by construction: an example is TF-correct iff argmax equals the truth
   at every answer position given the true prefix — and then autoregression feeds
   exactly those correct tokens back, reproducing the same string, so it is
   AR-correct too. Exact match is all-or-nothing, so any TF error also fails AR.
   (Teacher forcing would only flatter the model under *partial/per-token* credit,
   which this verifier does not give.)
2. **Argmax depth ≈ mean sampled depth (±0.5), and the sampled distribution is
   itself flat across difficulty.** Collapsing the router to its mode at inference
   hides nothing — there is no difficulty-dependent depth policy underneath.

**So the real gap was never train-vs-inference.** The drop from training reward
(~0.6, on *seen* train instances, with memorisation) to val (~0.11, *unseen*) to
extrap (0.000, *harder*) is the **generalisation** axis — seen → unseen → longer —
which is exactly the bottleneck identified in Experiments 1–3. The one genuine
forward-pass difference that remains (sampled depth in training, argmax at
inference) is empirically benign here: argmax ≈ mean and results are stable.

---

# Experiment 5 — the anchor variant (span-anchor subquadratic attention)

`anchor_rdr.py` implements the ANCHOR formalization (span landmarks, anchor score,
local-window ∪ top-k-landmark sparse attention with one exact softmax, Gumbel-TopK
selection, optional depth-persistent landmarks). Since short final-value tasks give
span selection nothing to select, it is tested on a **long-context KEYVAL retrieval**
task: many `key:val` pairs then `GET key =`, answer is that key's value; difficulty
= number of pairs = context length = needle distance (train 2..10, extrap 11..16,
sequences up to 77 tokens). Substrate attention is compared across
{linear, full, anchor, anchor+depth-residual}, 2500 steps each.

> ⚠️ **Bug found and fixed first.** linear and full produced *bit-identical*
> training curves — because the recurrent core started from `e` (the prelude
> output) and **discarded the substrate output `x`**, inherited from the original
> `rdr.py`. The whole sequence substrate (and any attention in it) was dead code
> w.r.t. the output. Fixed in `anchor_rdr.py` by starting the core from `x`. (The
> same latent bug still sits in `rdr.py`/`kaggle_rdr.py` — noted, not yet changed.)

## Results (KEYVAL retrieval accuracy, autoregressive + exact verify)

| Substrate attention | train reward (TF) | **interp** | **extrap** |
|---------------------|:---:|---:|---:|
| linear              | ~0.40 (plateaus)  | **0.310** | 0.225 |
| full                | ~0.40 (plateaus)  | **0.310** | **0.230** |
| anchor              | **1.000** (ce→0)  | 0.190 | 0.105 |
| anchor + depth-res  | **1.000** (ce→0)  | 0.095 | 0.125 |

## Reading (a clean, counter-intuitive result)

1. **The anchor mechanism works — almost too well.** It is the only variant that
   *solves training*: exact-match reward climbs to **1.0** and CE to **0** by
   ~step 750, while linear/full plateau near 0.40 and never fit the data. The
   span-anchor attention genuinely lets the model pick out the right pair.
2. **But it overfits and generalises worse.** Despite memorising the training set
   perfectly, anchor reaches only **0.19** interp / **0.11** extrap — *below* the
   "weaker" linear/full (0.31 / 0.23). Adding depth-persistent landmarks makes it
   worse still (0.095 / 0.125): more capacity → more memorisation, not more
   generalisation. The precise, high-capacity retrieval head fits specific training
   sequences instead of inducing the general "look up the queried key" rule.
3. **At this scale full ≈ linear.** Full attention's range advantage does not show
   up (both 0.31) — the bottleneck is representational capacity, not attention
   span, so this toy regime cannot reward the anchor mechanism's actual strength
   (cheap long-range access). Both linear and full also degrade with more pairs
   (linear d2 0.46 → d10 0.05), i.e. nobody truly learns scalable retrieval here.

### Bottom line

The anchor variant is **mechanistically real** (it alone fits the retrieval task)
but at toy scale its capacity is a liability: it memorises rather than generalises,
so on held-out data it *loses* to plain attention. This mirrors the whole study's
theme — the bottleneck is **generalisation**, and adding a more powerful mechanism
without more data/regularisation/scale moves memorisation, not generalisation. A
fair test of the anchor mechanism's intended benefit (subquadratic long-range
retrieval) needs (a) a regime where attention *span* is the bottleneck (much longer
context than 77 tokens, where full attention is actually expensive) and (b) enough
data/regularisation that the precise head cannot just memorise — i.e. the Kaggle
GPU scale-up, not CPU toy scale.

Artifacts (`*.jsonl`, `*_results.json`, `*.log`) are git-ignored; regenerate with
`python anchor_rdr.py` (see `README.md`).
