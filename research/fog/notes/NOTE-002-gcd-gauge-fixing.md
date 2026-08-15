# NOTE-002 — GCD law for periodic latent gauge ambiguity

Consider a semantic transition `S_u` parameterized by instruction/operand `u`
and a shared latent gauge transform `G` such that `G` commutes with all semantic
transitions in the tested family.  The implemented step is

`T_u = G S_u`.

After a depth-`d` program,

`T_{u_d} ... T_{u_1} = G^d S_{u_d} ... S_{u_1}`.

If only the terminal state is required to be canonical, then an exact solution
at depth `d` is compatible with any gauge satisfying

`G^d = I`.

For a set of supervised depths `D`, exact compatibility requires

`G^d = I` for every `d in D`.

Let `g = gcd(D)`.  By Bezout's identity, for invertible `G` this implies

`G^g = I`.

Therefore the exact residual periodic gauge has order dividing `g`.

Consequences:

- training only depth 2 can hide an order-2 chart;
- training only depth 3 can hide an order-3 chart;
- depths `{2,4,6}` still permit order 2 because gcd is 2;
- any set with gcd 1 removes nontrivial exact finite-order gauges in this
  commuting family.

This is a **gauge identifiability** statement, not an optimizer theorem.  EXP-003
explicitly shows a noncanonical local minimum for the coprime objective from a
far initialization.

Research implication for FOG: variable-depth training should not be chosen only
for curriculum convenience.  Depth-set arithmetic can determine whether a
periodic internal chart is even identifiable.
