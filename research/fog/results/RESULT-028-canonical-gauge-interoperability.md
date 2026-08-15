# RESULT-028 — Independently trained latent machines become interoperable after canonicalization

For all seed pairs 73->74, 73->75 and 74->75:

- canonical codebook mean cosine > `0.9999999`;
- minimum identity cosine > `0.99999986`;
- direct zero-shot transfer of all 31 states: **100%**;
- bridge imaginary residual: ~`1e-15`;
- 32 source steps + structural bridge + 32 target steps: **100%**.

No bridge parameters were trained.  The transform is derived from each model's
learned operator spectrum plus one common identity anchor.

Interpretation: the earlier failure of an unconstrained learned chart bridge is
not a general impossibility result.  When independently trained modules realize
the **same operator algebra**, a canonical gauge can make their latent states
interoperable without paired-state bridge training.

Boundary: this experiment relies on a simple-spectrum anchor and a shared
identity anchor.  Degenerate spectra and unrelated operator algebras remain
open.

Evidence: `artifacts/research/exp_029/metrics.json`.
