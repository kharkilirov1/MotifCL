# IDEA-001 — Latent Register Machine

Status: IDEA / parent direction
Evidence: E0 hypothesis

## Claim to investigate

Reasoning may be better modeled as repeated transitions of a structured latent machine state than as hidden natural-language sentences.

Instead of treating K latent slots as K vague "thoughts", assign machine-like roles:

`S_t = (address_t, value_t, control_t, scratch_t)`

and learn/construct a shared transition operator:

`S_{t+1} = T_theta(S_t, context)`.

The number of transitions becomes a compute budget independent of parameter count.

## Why this differs from latent CoT

Latent CoT often keeps the conceptual unit "a hidden sentence/step". The register-machine view instead asks whether the state is an executable interface:

- a value can become the next address;
- a control register can choose an operator/phase;
- scratch space can hold non-lexical intermediate computation;
- the same transition operator can be iterated beyond trained depth.

## First falsifiable consequence

If this view is useful, a minimal recurrent binder should compute `f^R(x)` on unseen permutation tables when `R` is supplied only by iteration count. A static-query system with identical parameters must fail for `R>1`.

See EXP-001.
