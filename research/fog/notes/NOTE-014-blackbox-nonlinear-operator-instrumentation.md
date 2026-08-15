# NOTE-014 — from black-box recurrence to structural evidence

EXP-035 suggests a production instrumentation pattern that does not require
reading an operator directly from model weights.

Given a recurrent hidden map `F_g` and a collection of visited states:

1. **probe topology** — apply candidate recurrent actions and identify recurring
   local regions / graph edges;
2. **probe derivatives** — estimate local Jacobians with finite perturbations or
   automatic differentiation;
3. **remove local gauge** — synchronize Jacobian fields into shared candidate
   actions when possible;
4. **measure gauge-invariant loops** — products around closed latent loops expose
   relations independent of local coordinate choices;
5. **compile conservatively** — project onto a structural law only if commutant,
   closure, gain or holonomy evidence crosses a fixed gate;
6. **fallback** — keep the original neural transition where the gate abstains.

The important shift is conceptual:

> the structural compiler need not know what a latent state *means*; it can infer
> reusable computation from how neighborhoods of latent states transform.

For a production transformer/recurrent backbone, local regions will not form an
exact finite orbit and Jacobians will be high-dimensional.  The next challenge
is therefore to replace exact orbit deduplication with local clustering/manifold
models and replace full Jacobians with low-rank/JVP sketches.
