# Mechanistic experiment report v2

Machine-readable results are in [the experiment matrix](artifacts/mechanistic_matrix/metrics.json).

## Bottom line

The strict latent recurrence has a real positive result, but the architecture is not yet robust enough to call a 10M geometry validated.

- One fixed-operator run reached 100% on trained depths and 75% on unseen depths.
- A second run reached 95.83% / 43.75%.
- A third run collapsed to chance.
- A new dynamic operator could not be applied even once: all one-hop probes stayed at chance.
- The selected K=4 checkpoints have almost collinear slot directions.
- The exploratory K sweep is not paired by initialization and cannot identify an optimal slot count.

The scientific verdict is therefore **NO-GO for a validated 10M architecture freeze**. Building and packaging a 10M experimental candidate is still reasonable if it is explicitly labelled unvalidated.

## 1. Fixed-operator recurrence

The prompt is `(operator, start state)` and omits hop count. The target is obtained by repeatedly applying one of the shifts `(1, 2, 3) mod 8`. The requested depth is communicated only through `R=L`. The strict final decoder receives a neutral `TASK` token plus latent memory, not the original prompt.

Shared geometry:

| field | value |
|---|---:|
| vocabulary | 15 |
| `d_model` | 48 |
| backbone layers / heads | 1 / 4 |
| backbone / planner FFN | 96 / 96 |
| latent / memory slots | 4 / 4 |
| compare rank | 12 |
| modes | 3 |
| parameters | 60,675 |

Training used AdamW with learning rate `3e-3`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-2`, gradient clipping at 1.0, and no scheduler. The curriculum used only depth 1 before step 300, depths 1–2 before step 800, and depths 1–4 thereafter.

### Observed run sensitivity

| run | model/data seed | steps | selected | ID depths 1–4 | OOD depths 5–8 |
|---|---:|---:|---:|---:|---:|
| strict seed 0 | 0 / 1101 | 2,200 | 1,000 | 96/96 = **100.00%** | 72/96 = **75.00%** |
| strict seed 1 | 1 / 2101 | 1,200 | 1,200 | 92/96 = **95.83%** | 42/96 = **43.75%** |
| strict seed 2 | 2 / 3101 | 1,200 | 1 | 13/96 = **13.54%** | 12/96 = **12.50%** |

This is preliminary run-seed sensitivity, not a clean robustness estimate: both model and data seeds changed, and seed 0 ran longer. It is nevertheless enough for a no-go because a complete chance-level collapse occurred.

The seed-0 OOD curve was 87.50%, 83.33%, 75.00%, and 54.17% at depths 5–8. For seed 1 it was 62.50%, 41.67%, 33.33%, and 37.50%.

### Memory and depth interventions

| selected checkpoint | normal ID | zero memory ID | shuffled memory ID | correct-R OOD | forced R=1 OOD | forced R=4 OOD |
|---|---:|---:|---:|---:|---:|---:|
| strict seed 0 | 100.00% | 12.50% | 0.00% | 75.00% | 8.33% | 8.33% |
| strict seed 1 | 95.83% | 12.50% | 1.04% | 43.75% | 9.38% | 8.33% |
| strict seed 2 | 13.54% | 12.50% | 10.42% | 12.50% | 13.54% | 12.50% |

The two successful checkpoints genuinely require example-specific memory and the correct iteration depth. The failed checkpoint provides the expected negative control.

### Historical non-strict run

An earlier full-prompt-decoder run reached 98.96% ID, but retained 81.25% after shuffling memory between examples and achieved only 9.38% OOD. This is consistent with a shortcut in which the prompt carries `(operator, state)` while memory mainly carries a depth code.

This is **not** a controlled bottleneck-only comparison. The historical model has 58,323 parameters, predates the final context-query change, and used a different code snapshot. It is evidence that the shortcut can occur, not a causal measurement of the bottleneck change alone.

## 2. Dynamic one-hop operator lookup

The next probe isolates the earlier failure before recurrence. Every example contains a freshly shuffled 8-state permutation and asks for exactly one transition. The serialized prompt is:

```text
[DB, dst(source=0), ..., dst(source=7), QUERY, start, STEPS, length=1, END]
```

Training used 1,000 steps, batch size 128, model seed 0, training base seed 1101, AdamW `3e-3`, and 4 CPU threads. Evaluation used 2,048 separately generated examples from base seed 99001. Chance is 12.5%.

| decoder | backbone layers | parameters | final CE | unseen accuracy |
|---|---:|---:|---:|---:|
| strict, one `DB` token | 1 | 65,811 | 2.082833 | 244/2,048 = **11.91%** |
| full original prompt | 1 | 65,811 | 2.081842 | 244/2,048 = **11.91%** |
| strict | 2 | 84,675 | 2.083156 | 242/2,048 = **11.82%** |
| strict | 3 | 103,539 | 2.082477 | 242/2,048 = **11.82%** |

All variants failed. Adding layers does not repair the current implicit-position table interface. A proper next gate should serialize explicit source/destination key-value rows and include a matched direct-Transformer baseline. Multihop dynamic training should not start until unseen one-hop accuracy is at least 90% across three seeds.

## 3. Exploratory latent-slot sweep

These runs changed only `latent_slots=K` on the strict fixed-operator task. K=1, 2, and 8 ran for 1,200 steps with data seed 1101; K=4 is the existing 2,200-step reference.

| K | model seed | steps | parameters | selected | ID | OOD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1,200 | 60,531 | 1 | 12/96 = 12.50% | 0/96 = 0.00% |
| 1 | 2 | 1,200 | 60,531 | 1 | 12/96 = 12.50% | 12/96 = 12.50% |
| 2 | 0 | 1,200 | 60,579 | 1 | 12/96 = 12.50% | 12/96 = 12.50% |
| 4 reference | 0 | 2,200 | 60,675 | 1,000 | 96/96 = 100.00% | 72/96 = 75.00% |
| 8 | 0 | 1,200 | 60,867 | 1 | 12/96 = 12.50% | 12/96 = 12.50% |

This table must **not** be read as proof that K=4 is uniquely optimal:

1. Changing the shape of the early learned query consumes a different number of RNG draws. Later tensors therefore receive different initial values even under the same integer seed.
2. K=4 ran for a different total number of steps.
3. Probe pairs shared one 9-CPU host, so their wall-clock times are not comparable throughput measurements.

A valid sweep needs name-stable initialization or loading all shape-compatible tensors from a common initial state, followed by multiple seeds.

### Slot-direction diagnostic

On exhaustive depth-4 prompts, the mean off-diagonal cosine similarity across the four recurrent steps was:

| checkpoint | selected step | mean cosine | per-step range |
|---|---:|---:|---:|
| strict seed 0 | 1,000 | 0.999920 | 0.999915–0.999930 |
| strict seed 1 | 1,200 | 0.999970 | 0.999958–0.999982 |
| strict seed 2 | 1 | 0.999659 | 0.999490–0.999890 |

The slots are nearly collinear in direction. Their small differences and different sequence positions can still matter—the failed K=1/2 runs show that simply deleting positions is not harmless—but the present evidence does not establish four independent parallel thoughts.

## 4. Unrun matrix cells

The following requested cells remain `n/a`; no numbers are manufactured for them:

- paired slot sweep;
- compare-rank sweep;
- memory-capacity sweep;
- conditioning ablation;
- explicit key-value dynamic operator lookup;
- dynamic multihop generalization after the one-hop gate.

These should follow trainer stabilization. Resource conclusions drawn before that would mostly measure optimization lottery.

## 5. Go/no-go and acceptance gates

Verdict: **NO-GO for declaring the 10M architecture validated.**

It is a **GO for an experimental 10M candidate and real-data pipeline**, provided checkpoints and documentation retain that label.

Before validation, require:

1. A controlled five-seed fixed-task run with fixed data seed and steps: at least 4/5 seeds at ID ≥95%, worst seed ≥90%, and median OOD ≥50%.
2. Unseen dynamic one-hop accuracy ≥90% across three seeds.
3. After that gate passes, unseen-table multihop ID ≥80% and depths 5–8 OOD ≥40%.
4. For successful strict runs, zeroing or cross-example shuffling memory must lower accuracy by at least 50 percentage points.
5. K, compare rank, and memory capacity must be selected using paired initialization and a performance/compute Pareto rule.

