# NOTE-004 — Why addition and multiplication create multi-chart pressure

For a prime field, translations and non-zero scalings satisfy

`M_a T_b M_a^{-1} = T_{ab}`.

In general `M_a T_b != T_b M_a`; together they form a non-abelian affine group.
A unitary basis can simultaneously diagonalize commuting normal operators, but
not an arbitrary non-commuting family.

The additive Fourier basis diagonalizes translations.  A log-Fourier basis on
`F_p*` diagonalizes multiplication in exponent coordinates.  There is no reason
to expect one basis to make both families equally local/diagonal.

EXP-008 is a finite controlled manifestation of this pressure, not a theorem
that no other representation can make both operations efficient.  The stronger
research question is a complexity frontier:

> what is the minimum operator complexity required to support a chosen family
> of non-commuting transformations in one latent chart?

This suggests measuring operator rank / sparsity / local tensor complexity as a
first-class architectural budget, rather than only latent dimension.
