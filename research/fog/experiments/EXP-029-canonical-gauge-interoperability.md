# EXP-029 — Canonical gauge and zero-shot module interoperability

Status: **PASSED**  
Date: 2026-08-14

Independently train d=30 representations on seeds 73/74/75.  For each model:

1. recover A's spectral gauge;
2. sort eigenspaces by nearest finite-order root;
3. use only the shared identity state `E(0)` to fix per-mode complex phase.

No bridge network is trained and no paired non-anchor identities are used to
fit the bridge.

Tests:

- compare canonical codebooks across seeds;
- transfer all latent identities from one original gauge to another;
- execute 32 recurrent steps in a source model, bridge the continuous state,
  then execute 32 more steps in a target model.
