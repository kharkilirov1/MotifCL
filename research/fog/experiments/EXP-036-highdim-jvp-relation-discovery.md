# EXP-036 — High-dimensional JVP relation discovery

## Question

Can the structural compiler test recurrent operator laws at production-like hidden
width without materializing a full Jacobian?

## Protocol

A nonlinear black-box map hides the nontrivial affine representation of
`F_31` inside 128D, 256D and 320D observed coordinates.  The compiler receives
only function calls.  Candidate laws are searched cheaply in state space and
only the best candidates are verified with a small number of randomized JVPs.

Laws:

- `A^31 = I`;
- `M^30 = I`;
- `M A M^-1 = A^3`.

A 32D arm additionally computes the complete Jacobian as an oracle-only check.
Perturbation arms deliberately break the exact algebra.

## Acceptance

- exact system: recover all three laws;
- JVP relation residual agrees with the full-Jacobian oracle in 32D;
- 128/256/320D operate without full Jacobians;
- perturbation should increase both state and JVP residual rather than produce a
  false zero-evidence claim.

## Result

**PASS.**

- 128D and 256D recover `(31,30,3)` at perturbations 0, 1%, 3% and 5%; residuals
  increase smoothly with perturbation.
- 320D at 3% perturbation also recovers `(31,30,3)` using 8 JVP samples per
  accepted relation.  State/JVP residuals are about 0.11--0.13.
- the 32D exact oracle gives numerical-zero full-Jacobian and JVP residuals.
- a real ~10M FOG transition with `d_model=320`, `K=4` is JVP-probeable after
  locally forcing PyTorch's differentiable math-SDPA kernel for the probe only.

Artifacts: `artifacts/research/exp_036/`.
