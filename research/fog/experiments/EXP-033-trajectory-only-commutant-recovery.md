# EXP-033 — trajectory-only structural recovery

Status: **PASSED controlled**  
Date: 2026-08-14

## Question

Can the compiler recover repeated latent operator structure without direct
access to action matrices or discrete latent identities?

## Protocol

For each hidden action, collect only continuous transition pairs

\[
z \rightarrow f(z)
\]

from random normalized latent probes.  Add output-state noise and estimate one
dense action by ridge linear regression.  Then run the EXP-032 approximate
commutant compiler.

The identification/compiler stages receive:

- no canonical codebook;
- no discrete identity labels;
- no true action matrices;
- no multiplicity metadata.

The codebook is used only by the held-out evaluator after compilation.

Sweep:

- multiplicity 3 and 4;
- seeds 130,131,132;
- trajectory counts `d`, `2d`, `4d` per operator;
- output noise 5% and 10%;
- depth 256.

Artifact: `artifacts/research/exp_033/metrics.json`.
