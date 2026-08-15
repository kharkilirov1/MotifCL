# EXP-032 — approximate commutant and repeated-block denoising

Status: **PASSED controlled**  
Date: 2026-08-14

## Question

Does the commutant signature survive approximate learned operators strongly
enough to identify repeated structure and denoise recurrent dynamics?

## Protocol

Start from EXP-031, perturb each dense action by relative Gaussian matrix noise,
then project the perturbed matrix to its nearest orthogonal matrix.  The
compiler is **not given multiplicity**.

It examines the singular spectrum of the joint commutator constraint and tests
square candidate low-singular subspaces `m^2`.  A repeated-irrep hypothesis is
accepted only when the next singular value is at least 5x larger than the top
singular value inside the candidate subspace.

After acceptance:

- approximate invariant copies are split and aligned;
- corresponding blocks are averaged;
- a separate optional arm projects that average to the nearest orthogonal block.

Generic random orthogonal operator pairs are negative controls.

Sweep:

- multiplicities 2,3,4;
- seeds 120,121,122;
- relative noise 3%, 5%, 10%, 15%;
- depth 256, 2,048 trajectories.

Artifact: `artifacts/research/exp_032/metrics.json`.
