# EXP-001 — Recurrent Binding Composition

Status: PROTOCOL FROZEN (validation phase)
Date frozen: 2026-08-14
Parent: IDEA-001
Target evidence: E3 composition

## Research question

Can an exact/near-exact protected payload from hop `t` become the address for hop `t+1`, allowing the same latent operator to compute `f^R(x)` without decoding intermediate states?

## Task

Each example contains a shuffled table for an 8-state permutation `f` and a start query `x`.

Target at recurrent depth R:

`y = f^R(x)`.

The mapping itself is hash-partitioned into train/validation/test, so the same permutation cannot reappear in another split under a different row order or query.

The prompt does not contain R. R exists only as the number of latent transitions.

## Representation gate

The first decisive gate uses one frozen canonical state codebook for both source identities and payload identities. Role is supplied by the computation path, not by unrelated `KEY_*` versus `VALUE_*` embeddings.

This intentionally isolates composition from the separate typed-role problem.

## Arms

1. `recurrent`: `query_{t+1} = primary_t`.
2. `static`: every hop reuses `query_0` (current production-v2 failure mode).
3. `hard_recurrent`: nearest canonical identity is fed back each hop; diagnostic ceiling only, never a promoted mechanism.

All arms share the same state codebook, binder and readout.

## Training

Only address sharpness is trainable in the minimal gate. Train on mapping-disjoint train examples and depths 1–4. No intermediate state supervision.

This asks whether a single scalar precision parameter is enough for the recurrent mechanism; later experiments may train richer address transforms, but they require new protocol IDs if they change the decisive claim.

## Validation

Evaluate depths 1–4 (ID depth) and 5–16 (OOD depth) on validation mappings.

For every depth record:

- exact final accuracy;
- NLL;
- per-hop correct-address top1;
- per-hop correct-address probability mass;
- address entropy;
- cosine of the primary state to the oracle canonical state.

## Causal intervention

For depths >=3, after hop 2 roll the recurrent query across batch examples while keeping each example's table fixed. Downstream accuracy should collapse relative to normal recurrence.

## Pass criteria before locked test

The locked test may be opened only if all are true on validation:

1. recurrent accuracy >= 99% for each depth 1–4;
2. recurrent mean accuracy >= 95% for depths 5–12;
3. static arm is materially worse than recurrent for depths 2–4;
4. hop-2 query corruption reduces depth>=3 accuracy by at least 50 percentage points or to near chance;
5. trajectory metrics do not show hidden catastrophic drift before depth 12.

If `hard_recurrent` passes while soft `recurrent` fails, classify as coordinate-stability failure and move to IDEA-002 rather than scaling the model.

## Locked-test rule

Do not touch the `test` split while changing architecture/hyperparameters. Once opened, any subsequent mechanism change receives EXP-002 or later.
