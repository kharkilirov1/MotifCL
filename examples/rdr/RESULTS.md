# RDR mechanism study — training run

Full RDR model trained on the bounded verifiable-task dataset (3 families,
difficulty == reasoning depth). This is a **mechanism probe at toy scale**, not a
capability result.

## Setup

| | |
|---|---|
| Model | `d_model=96`, `n_recur=6`, full RDR (recurrence + AttnRes + RLVR) |
| Params | **755.8K** |
| Dataset | `build_dataset.py` defaults — train 3801 / val_interp 647 / test_extrap 2400 |
| Train | 3000 steps, batch 48, Adam lr 2e-3, ponder 0.01, exact-verifier RLVR |
| Eval | **real autoregressive generation** scored by exact `verify()` |
| Wall time | ~26 min, CPU (4 cores) |

## Training curve (teacher-forced exact-match reward)

```
step    1 | ce 5.68 | reward 0.00 | depth 3.5
step  600 | ce 0.96 | reward 0.31 | depth 2.7
step 1800 | ce 0.91 | reward 0.40 | depth 3.0
step 2200 | ce 0.77 | reward 0.54 | depth 3.1
step 3000 | ce 0.62 | reward 0.60 | depth 3.3
```

The exact-verifier reward climbs from 0 → ~0.6: the RLVR loop learns to produce
*fully correct* answers, not just the right format.

## Final evaluation (autoregressive generation + exact verify)

| Split | Accuracy | by family | by difficulty |
|-------|---------:|-----------|---------------|
| `val_interp` (seen difficulties, unseen instances) | **0.223** | modchain 0.12 · ptrchase 0.27 · eval 0.32 | d1 0.18 · d2 0.34 · d3 0.23 · d4 0.16 |
| `test_extrap` (**harder** than any training example) | **0.147** | modchain 0.10 · ptrchase 0.26 · eval 0.06 | d5 0.10 · d6 0.19 |

Chance is effectively ~0 for exact string match on multi-token answers, so both
splits are clearly above chance. Accuracy is lower than the teacher-forced reward
(~0.6) because autoregressive decoding compounds per-token errors.

## Mean router depth per difficulty

| difficulty | 1 | 2 | 3 | 4 | 5 (extrap) | 6 (extrap) |
|------------|---|---|---|---|---|---|
| mean depth | 3.60 | 3.18 | 3.42 | 3.56 | 3.44 | 3.38 |

## Reading of the result (honest)

1. **The loop × exact-reward wiring works.** Training reward rises 0→0.6 and the
   model produces exactly-correct answers; nothing in the supervised+RLVR pipeline
   is broken.
2. **The mechanism transfers above chance to strictly harder problems**
   (`test_extrap` 0.147), strongest on `ptrchase` (0.26) where extra hops are
   pure pointer following — the cleanest recurrence-shaped task.
3. **The central hypothesis is NOT clearly supported in this run.** Mean router
   depth is essentially **flat (~3.2–3.6) across difficulty** and does not grow on
   the extrapolation difficulties. At this scale, with single-sample REINFORCE and
   a batch-mean baseline, the router has not discovered a strong "harder ⇒ deeper"
   policy. This is the interesting negative signal, and it is exactly what this
   probe is designed to surface.

### Plausible reasons the depth signal stays flat (next experiments)

- Reward advantage uses a **global batch-mean baseline**; a per-difficulty (or
  per-family) baseline would give the router a cleaner credit signal for *when*
  depth helps.
- Single-sample REINFORCE is high-variance — multiple rollouts / a value baseline
  would tighten it.
- The `ponder` penalty pushes uniformly toward less depth; a curriculum or a
  difficulty-aware ponder schedule would let depth specialise.
- Run the **section-6 ablation grid** (`train.py --ablate`) to check whether
  removing recurrence / AttnRes / RLVR actually hurts extrap accuracy — that
  isolates each component's contribution.

Artifacts (`rdr_dataset.jsonl`, `rdr_results_full.json`, `train_full.log`) are
git-ignored; regenerate with the commands in `README.md`.
