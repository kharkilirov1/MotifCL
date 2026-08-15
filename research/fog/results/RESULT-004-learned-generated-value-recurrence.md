# RESULT-004 — Learned one-step ALU composes to depth 32 on OOD transitions

The Fourier/local operator was learned only from one-step train pairs.  During
recurrent evaluation every individual transition was deliberately selected
from the held-out pair split.

Across all three split seeds:

| depth | Fourier/local | Fourier/full | random/full |
|---:|---:|---:|---:|
| 1 | **100%** | 59.39% | 3.08% |
| 2 | **100%** | 0.18% | 2.32% |
| 4 | **100%** | 0.03% | 1.78% |
| 8 | **100%** | 0.18% | 1.81% |
| 16 | **100%** | 0.08% | 1.79% |
| 32 | **100%** | 0.20% | 1.78% |

For Fourier/local, every intermediate hop also remained 100% canonical with
minimum cosine effectively 1.0.  No intermediate state was converted to a
symbolic value between steps.

This crosses an important controlled boundary beyond retrieval:

1. the operator creates a new latent value;
2. that value is not selected/copied from a prompt table;
3. the generated continuous state becomes the next computation input;
4. the same learned operator repeats through unseen depth;
5. every actual transition is OOD relative to the one-step training pairs.

The full Fourier arm is an instructive negative control: even moderate one-step
held-out accuracy is not enough for recurrence.  Small coordinate errors become
catastrophic under feedback when the learned rule is not the true shared law.

Limit: this is modular arithmetic in a deliberately compatible Fourier chart,
not yet a learned natural-language reasoner or production 10M result.

Evidence: `artifacts/research/exp_005/metrics.json`.
