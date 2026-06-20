# RDR — Recurrent-Depth Reasoner (mechanism study)

A small, self-contained reference that wires up the RDR architecture and trains it
on a bounded set of **verifiable** reasoning tasks, where **difficulty == reasoning
depth**. The point is not scale or frontier capability — it is to test, at toy
scale on CPU, whether the central mechanism is real:

> Does a token-level **adaptive-depth recurrent loop**, shaped by an **exact
> verifier reward** (RLVR) and aggregated with **attention-over-depth (AttnRes)**,
> allocate *more* recursion to *harder* problems — and does that transfer to
> problems strictly harder than anything seen in training (systematic
> generalisation)?

## Files

| File | Role |
|------|------|
| `rdr.py` | The model: sub-quadratic substrate (+full-attn recall layer), recurrent reasoning core with a MoR depth router, depth-wise LoRA, AttnRes, MoE prelude/coda, MTP head. Ablation flags: `use_recurrence` / `use_attnres` / `use_rlvr`. |
| `build_dataset.py` | 3 discrete families with **exact** ground truth (`modchain`, `ptrchase`, `eval`). Difficulty = number of hops / nesting. Splits: `train`, `val_interp` (unseen, same difficulty), `test_extrap` (**harder** than train). |
| `train.py` | Char-level trainer. Supervised next-token CE on the **answer span only**; RLVR REINFORCE using the **exact verifier** (`build_dataset.verify`) as the reward that shapes the router's per-token depth. Evaluation is **real autoregressive generation** scored by `verify()`. |

## Quickstart

```bash
pip install torch

# 1) build the dataset (exact, deduplicated, difficulty-stratified)
python build_dataset.py --out rdr_dataset.jsonl

# 2) train the full RDR model
python train.py --data rdr_dataset.jsonl --steps 3000 --batch 48

# 3) (optional) run the section-6 ablation grid
python train.py --data rdr_dataset.jsonl --steps 3000 --ablate
```

## How the pieces connect

- **Tokenisation**: byte/char level. Specials live above the byte range so they
  never collide with real characters: `PAD=256, BOS=257, EOS=258` → `vocab=259`.
- **Supervision**: CE is masked to the answer region (`target` chars + `EOS`); the
  prompt is never used as a target.
- **Reward (RLVR)**: `exact_reward` is the tensor analogue of `build_dataset.verify`
  — reward `1` iff the model's greedy answer matches the *entire* target, else `0`.
  This binary, executable-style signal is what the REINFORCE term uses to push the
  MoR router toward useful depth choices. A `ponder` penalty discourages
  over-looping.
- **Measurement**: at eval we log accuracy and **mean router depth per difficulty**
  on both the interpolation and extrapolation splits — that is the direct test of
  the hypothesis.

## What to look for in the output

- `reward` climbing during training (exact answers, not just format).
- `depth/diff` increasing with difficulty `d` — the router spending more recursion
  on harder instances.
- `test_extrap` accuracy above chance, and the depth trend continuing into the
  unseen harder difficulties — evidence the *mechanism* (not memorisation)
  transfers.

## Run on a Kaggle GPU (moderate scale-up)

`kaggle_rdr.py` is a **single self-contained file** (model + dataset + exact
verifier + trainer + eval, all inline, CUDA-ready) for a free Kaggle T4. It trains
two models at the same moderate scale — **full RDR** vs a **depth-1 baseline** —
and prints an interp/extrap comparison, so you can see whether the recurrent-depth
mechanism earns its keep at larger scale.

1. New Kaggle Notebook → Settings → Accelerator → **GPU T4 x1**.
2. Paste the whole file into one cell (or add it as a script), **Run All**.
3. Read the `=== SUMMARY ===` block at the end; full results land in
   `kaggle_rdr_results.json`.

Scale lives in the `CONFIG` block at the top (`D_MODEL`, `N_RECUR`, `BATCH`,
`STEPS`, dataset `D_TRAIN`/`D_MAX`). Lower `BATCH`/`D_MODEL` if you hit OOM. The
default (`d_model=256`, `n_recur=8`, `batch=128`, `12000` steps) finishes well
inside one free session.

This is a **mechanism probe**, calibrated to a small from-scratch model. For scale,
swap the toy modules for the open implementations referenced in `rdr.py`
(MoR, Kimi-Linear/Attention-Residuals, Titans) and replace the stub verifier hook
with `pattern_generator.py`'s executable verifier.
