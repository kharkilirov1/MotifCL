# RESULT-036 — Finite operator grammar needs discrete forward routing

EXP-037 is the first direct mechanistic validation of the model-side
`register_machine_v3` cell.

The decisive change is not more parameters.  It is the combination:

`exact READ proposal -> typed value/control registers -> finite operator
candidates -> hard operator selection -> recurrent value-as-next-address`.

Soft mixtures of individually valid operators are generally not themselves a
valid operator family and accumulated geometric error.  Straight-through hard
routing preserves a discrete grammar in the forward pass while keeping router
gradients trainable.

A structured law-compatible primitive then remains exactly recurrent through
OOD depth 8 in the controlled task.  This does **not** establish natural-language
reasoning; it establishes that the production-shaped cell can perform generated
latent computation rather than only lookup.
