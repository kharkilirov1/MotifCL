# NOTE-006 — Operator-orbit width as a representation budget

Let additive characters over `F_p` be

\[
\chi_k(x)=\exp(2\pi i kx/p).
\]

Addition acts locally:

\[
\chi_k(x+y)=\chi_k(x)\chi_k(y).
\]

Scaling by `a != 0` acts on the frequency index:

\[
\chi_k(ax)=\chi_{ka}(x).
\]

Therefore a subset of frequencies `S` is closed under a set of allowed scalings
`A` only if

\[
k\in S, a\in A \Rightarrow ka\in S.
\]

For the full multiplicative group `F_p^*`, the action on non-zero frequencies
is transitive.  Hence the only non-empty subset of non-zero frequencies that is
invariant under **all** scalings is the full set of `p-1` non-zero frequencies.

More generally, if allowed scalings form a subgroup `H`, the orbit of one
non-zero frequency under `H` has size `|H|`.  EXP-020 verifies that a real
representation of width `2|H|` built from that orbit executes mixed ADD and
MUL-by-H programs exactly through depth 128.

This gives a controlled representation scaling law:

> **the width required by a sparse shared action can scale with the orbit of the
> basis under the operator family.**

It also exposes a design tradeoff:

- a wide invariant representation eliminates chart switching;
- compact operator-specialized charts reduce width but require CAST/bridge laws;
- partial operator families may admit much smaller invariant subspaces than the
  full grammar.

Evidence: `artifacts/research/exp_019/metrics.json` and
`artifacts/research/exp_020/metrics.json`.
