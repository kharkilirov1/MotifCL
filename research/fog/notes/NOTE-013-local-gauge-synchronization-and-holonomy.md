# NOTE-013 — local gauge fields, shared operators and holonomy

A nonlinear/state-conditioned latent system need not expose a single global
operator matrix.  If local coordinates vary with state, a shared logical action
can appear as

\[
J_{g,x}=H_{gx}R_gH_x^{-1}.
\]

Here `H_x` is a local gauge and `R_g` is the coordinate-independent operator.
The collection of local Jacobians is therefore a **cocycle** over the state
transition graph.

Two structural signals become available:

1. **gauge synchronization** — solve jointly for local `H_x` and shared `R_g`;
2. **holonomy** — multiply local Jacobians around a closed state loop.

For a loop `x_0 -> ... -> x_0`, the local gauges cancel:

\[
J_n\cdots J_1
=H_{x_0}(R_n\cdots R_1)H_{x_0}^{-1}.
\]

Thus loop products expose gauge-invariant operator relations even when no global
matrix exists in the observed coordinates.

EXP-034 uses this only in a very small O(2) system, but the principle is broader:
**closed latent trajectories can supply structural constraints to a compiler
without semantic decoding of the intermediate states.**
