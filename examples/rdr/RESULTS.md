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
