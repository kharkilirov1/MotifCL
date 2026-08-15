# RESULT-032 — hidden-state traces are sufficient for structural compilation in the controlled linear gate

EXP-033 establishes a practical trace-level interface to the compiler.

At approximately `2d` probes per operator:

- repeated structure was accepted on **12/12 runs**;
- raw identified depth-256 accuracy averaged **27.4%**;
- compiled shared-block accuracy averaged **99.988%**;
- worst compiled run was **99.85%**.

At `4d` probes all 12 runs compiled to 100%.

At only `d` probes, the current two-stage ridge -> commutant pipeline was not
reliably identifiable: only 1/12 arms crossed the fixed structural gap.  This is
a useful abstention regime, not evidence for a fundamental `2d` lower bound.
A joint structured system-identification method could require fewer probes.

## Interpretation

For the first time in this line, structural compilation works from **noisy
hidden-state transition traces** rather than privileged access to learned
operator matrices or semantic latent IDs.  This is directly compatible with a
future production diagnostic that logs backbone states around recurrent steps.
