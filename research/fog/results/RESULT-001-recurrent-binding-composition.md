# RESULT-001 — Recurrent Binding Composition

Status: PASSED / promoted to controlled E3 evidence
Experiment: EXP-001
Date: 2026-08-14
Locked test opened: yes (frozen checkpoint only)

## Result

A canonical protected payload can be reused directly as the next address and can therefore implement exact multi-hop latent composition without decoding intermediate states.

The model trained only one scalar (`bind.logit_scale`) on mapping-disjoint train examples at depths 1–4. Final learned scale: **2.4493382**.

### Validation

- recurrent final accuracy: **100% at every depth 1–16**;
- training depths: 1–4;
- OOD depths: 5–16;
- exploratory validation stress test: **100% at depths 24, 32, 48 and 64**;
- static-query baseline: 100% at depth 1, then strongly below recurrent (for example ~12.2% at depth 2);
- query corruption after hop 2: ~14.1% for all later tested depths.

### Locked test

The frozen checkpoint was evaluated on 1,024 mapping-disjoint test examples after all protocol criteria were satisfied.

- recurrent: **100% at every depth 1–16**;
- static: 100% / 12.99% / 26.86% / 26.56% at depths 1–4;
- recurrent with query rolled across examples after hop 2: **15.23%** at depths 3+;
- at depth 16, final-hop correct-address mass = **0.5101895** and oracle cosine = **0.9400254**.

A larger 4,032-example all-arm sweep was attempted but exceeded the execution limit before writing a result. This is classified `ENVIRONMENT`; it did not change the checkpoint or protocol. The completed locked evaluation uses 1,024 examples.

## Causal interpretation

The recurrent arm and static arm share the same frozen codebook, binder and readout. Their difference is only the recurrence rule:

- static: `q_{t+1} = q_0`;
- recurrent: `q_{t+1} = primary_t`.

The static arm therefore repeatedly computes the first hop, while the recurrent arm follows the permutation trajectory. Corrupting the state after hop 2 destroys the remaining trajectory, showing that later predictions depend on the example-specific intermediate latent state.

## Unexpected positive finding: bounded soft-state error

The recurrent state does not approach the exact one-hot code as depth grows, but it also does not diffuse indefinitely. Its geometry converges to a stable soft code:

- correct-address mass -> **0.510181...**;
- oracle cosine -> **0.9400218...**.

This is explained exactly by the symmetric fixed-point analysis in `research/notes/NOTE-001-soft-binding-fixed-point.md`.

## What this proves

Within the controlled canonical-code gate:

1. one protected latent output can be the next step's input;
2. the same differentiable operator composes repeatedly;
3. iteration depth can exceed training depth by at least 16x in the exploratory stress test (trained max 4, validated to 64);
4. intermediate states are causally necessary;
5. soft recurrent binding can have a stable non-discrete attractor rather than accumulating unbounded error.

## What this does NOT prove

- It is not yet the full 10M production path.
- It uses a canonical shared state codebook and a permutation lookup task.
- It does not generate a novel intermediate value absent from the table.
- It does not establish natural-language reasoning or an advantage over text CoT.
- It does not solve the role/identity issue created by separate key/value token namespaces.

## Decision

**Promote the recurrent-feedback mechanism to the next production-path experiment, not yet to the default architecture.**

Next: EXP-002 — production 10M recurrent token composition with backward-compatible `binding_query_update` and shared identity tokens.
