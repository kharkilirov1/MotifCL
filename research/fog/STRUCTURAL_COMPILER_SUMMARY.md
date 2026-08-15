# Structural Operator Compiler — current research summary

Updated: 2026-08-14

This document summarizes EXP-023 through EXP-035.  It does not replace the
individual frozen protocols or machine-readable artifacts.

## Core finding

A latent operator that looks dense in the training gauge may implement a very
small reusable computation in another gauge.  In the controlled affine setting,
a useful workflow is now:

```text
learn approximate invariant state/action system
    -> discover finite/operator relations
    -> recover or synthesize a canonical gauge anchor
    -> expose sparse motifs
    -> project onto the accepted legal operator family
    -> remove redundant operators
    -> execute recurrently / bridge compatible modules
```

## Evidence chain

### EXP-023 — gauge recovery

A learned dense d=30 action pair becomes diagonal/monomial in the automatically
recovered A eigengauge.  Hard pruning preserves 100% depth-64 execution.

### EXP-024 — recurrent denoising

Motif projection improves noisy long-horizon execution.  Adding the accepted
finite-order/norm law restores 100% in the tested noise range where raw dense
execution is near chance.

### EXP-025 — law discovery

The compiler infers from matrices alone:

\[
A^{31}\approx I,
\qquad MAM^{-1}\approx A^3.
\]

No field/Fourier semantic metadata is used by discovery.

### EXP-026 — grammar compression

Separately learned S3/S5/S7 modules compile to one primitive scaling generator:

\[
S5\approx S3^{20},\qquad S7\approx S3^{28}.
\]

Redundant dense matrices can be deleted without changing depth-64 behavior.

### EXP-027/028 — transition sample law

For a d-dimensional orthogonal action, k known independent directions leave an
`O(d-k)` completion freedom.  In d=30:

- k=28 leaves continuous O(2) ambiguity and is not repaired;
- k=29 leaves a +/- orientation ambiguity; motif structure repairs the wrong
  branch in the observed bad seed;
- k=30 determines the action directly.

### EXP-029 — module interoperability

Independent training seeds canonicalize to almost identical coordinates.  A
zero-parameter structural bridge transfers both static identities and
mid-program recurrent states at 100%.

### EXP-030 — synthesize a missing primitive

The trained library exposes only degenerate order-10 and order-3 actions.  The
compiler searches their composition closure and synthesizes an order-30,
simple-spectrum primitive that was never trained as its own module.  Original
operators are then recompiled as powers of the synthesized primitive.

### EXP-031 — repeated joint irreducible blocks

When multiplicity forces every operator/composition to have repeated spectrum,
the compiler switches from a spectral anchor to the **joint commutant**.  The
commutant dimensions 4/9/16 reveal multiplicities 2/3/4, symmetric commutant
elements split invariant copies, and intertwiners align them into one shared
2x2 grammar.  Depth-256 execution is 100% on all main runs.

### EXP-032 — approximate commutant denoising

Exact nullspaces are replaced by low-singular-value clusters.  A fixed
commutator-spectrum gap detects repeated structure through 15% action noise,
and shared-block averaging acts as structural error correction.

### EXP-033 — traces instead of matrices

The compiler can start from noisy continuous hidden-state transition pairs.
After ridge system identification, approximately `2d` probes per operator are
sufficient in the tested setting for robust commutant discovery and ~100%
depth-256 compiled execution.  No semantic codebook or latent identity labels
are used by identification/compilation.

### EXP-034 — state-conditioned local Jacobians

A single observed global action matrix is no longer required.  Local Jacobians
are treated as a gauge cocycle

\[
J_{g,x}=H_{gx}R_gH_x^{-1}.
\]

Gauge synchronization recovers shared actions.  Closed-loop holonomy supplies
gauge-invariant evidence for finite-order projection.  Through 10% Jacobian
noise, all tested runs return to 100% depth-256 continuous-state tracking after
the fixed holonomy gate accepts the law.  When the gate fails at the hardest
15% arm, the compiler abstains.

### EXP-035 — nonlinear black-box instrumentation

The compiler is no longer handed the context graph or Jacobians.  From one
hidden state it actively explores a nonlinear black-box orbit, infers graph
edges by black-box calls, estimates Jacobians by finite differences, and then
applies gauge synchronization plus holonomy-gated compilation.  All 12 main
arms through 10% Jacobian noise retain 100% depth-256 tangent tracking after
compilation.

## Current architectural hypothesis

FOG should not require every learned module to remain in the exact parameter
form used during gradient training.  A more promising architecture is:

1. **neural proposal layer** — learn approximate states/actions from data;
2. **structural compiler** — discover recurrently stable operator structure;
3. **compiled latent machine** — execute the simpler accepted grammar;
4. **fallback neural path** — retain uncompiled transitions where no structural
   hypothesis passes its residual/causal gates.

The compiler must be conservative: a motif is not accepted because it looks
simple.  It must satisfy fixed relation residuals and preserve held-out/recurrent
behavior.

## What remains open

The compiler no longer requires simple spectra, exact actions, direct action
matrices, or even one global observed matrix in the latest controlled gates.
Production FOG is still harder:

- local contexts/transition graphs must be inferred rather than supplied;
- Jacobians must be estimated from a genuinely nonlinear backbone rather than
  directly provided as local matrices;
- gauges may be general high-dimensional/non-orthogonal charts;
- semantic identity may be distributed rather than a fixed row codebook;
- the correct operator family may change by context;
- compilation may need to happen online or during training;
- approximate modules may not share a common anchor state.

## Next decisive experiment

Move from the finite 2D nonlinear orbit to a **continuous high-dimensional
hidden-state cloud**.  Discover neighborhoods without a known context count,
estimate only low-rank JVP/VJP sketches, and test whether shared operator motifs,
commutants and loop laws can still be recovered robustly.

This is now the closest controlled precursor to production instrumentation: the
remaining gap is primarily scale/manifold complexity rather than privileged
access to action matrices or semantic latent identities.
