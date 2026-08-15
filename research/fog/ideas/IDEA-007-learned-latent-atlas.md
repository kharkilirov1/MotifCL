# IDEA-007 — Learned latent atlas

The hand-designed Fourier results suggest a stronger architecture than anonymous
workspace slots:

`semantic identity/register -> choose chart -> apply local operator -> maintain/translate charts`.

The research goal is **not** to hard-code Fourier arithmetic into an LLM.  It is
to make the network discover operator-compatible charts from data while keeping
those charts canonical enough to be reused recurrently.

Possible mechanisms:

- multiple low-dimensional chart projections from a shared identity carrier;
- operator-conditioned equivariance/composition losses;
- joint diagonalization / low-commutator objectives for approximately commuting
  operator families;
- explicit chart-transition modules trained with cycle consistency;
- mixture-of-charts routing where the operator family selects the active chart;
- regularization on operator tensor rank/sparsity so geometry is rewarded for
  making useful transformations simple.

Primary failure mode to avoid: chart IDs or bridges becoming another memorized
finite lookup table.
