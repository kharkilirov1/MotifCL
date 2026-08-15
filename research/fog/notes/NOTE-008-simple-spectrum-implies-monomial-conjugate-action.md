# NOTE-008 — Simple spectrum turns conjugate actions into monomial motifs

Let `A` be diagonalizable with simple spectrum and let `M` satisfy

\[
M A M^{-1}=A^r.
\]

If `v` is an `A` eigenvector with eigenvalue `lambda`, then

\[
A^r(Mv)=MAv=\lambda Mv.
\]

Because the spectrum is simple, `Mv` must lie in the unique eigenspace whose
`r`-th eigenvalue power equals `lambda`.  Therefore in the eigenbasis of `A`,
`M` has at most one non-zero entry in every input eigenspace: it is monomial up
to per-mode complex scale.

For EXP-023, `A` is the learned translation generator and the discovered
relation is `M A M^{-1} ~= A^3`.  This explains why a dense learned `M` becomes
one-entry-per-mode after the gauge is recovered.

This statement is broader than the finite-field toy task: whenever a learned
operator has a well-separated simple spectrum and another operator normalizes
its generated algebra, spectral gauge recovery is a principled motif-discovery
mechanism.
