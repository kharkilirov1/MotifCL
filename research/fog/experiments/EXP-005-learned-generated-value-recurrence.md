# EXP-005 — Recurrent reuse of learned generated values

Status: **PASSED (controlled M4 gate)**  
Date: 2026-08-14

## Question

Can an operator learned only from one-step examples generate a continuous latent
value and then reuse its *own output* as the next input without intermediate
decoding, snapping or supervision?

## Protocol

The one-step operators from EXP-004 are frozen after fitting.  Runtime chains
are constructed so that **every transition pair `(current_value, operand)` is in
the held-out pair set used by EXP-004**.

At each hop:

`z_{t+1} = ALU(z_t, E(b_t))`.

`z_{t+1}` is normalized but never decoded to an integer and re-embedded before
the next step.

Depths: `1, 2, 4, 8, 16, 32`.

## Reproduction

```bash
python learned_generated_value_recurrence_experiment.py \
  --output artifacts/research/exp_005/metrics.json \
  --examples 2048
```
