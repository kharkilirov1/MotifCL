# NOTE-007 — A 30D barrier for exact linear affine registers over F_31

This note proves a dimension statement for the controlled EXP-021/022 setting.
It is **not** a lower bound for arbitrary nonlinear neural networks.

## Setup

Let `V = R^{F_31}` have basis `e_x` for the 31 field elements.  Let the two
permutations

\[
A:x\mapsto x+1,
\qquad
M:x\mapsto 3x
\]

act linearly on V.  Here 3 is primitive in `F_31^*`.

Suppose latent codes `E_x in R^d` and linear latent actions `A'`,`M'` satisfy
exact equivariance

\[
A'E_x=E_{x+1},\qquad M'E_x=E_{3x}.
\]

Define the linear map

\[
L:V\to R^d,\qquad L(e_x)=E_x.
\]

Then L intertwines the two actions.

## Constant and zero-sum subspaces

The constant vector spans a 1D invariant subspace.  Its complement

\[
W=\{v\in V:\sum_x v_x=0\}
\]

has dimension 30.

If `L` annihilated W, then

\[
L(e_x-e_y)=0
\]

for every x,y, so every latent identity code would be identical.  Therefore any
representation that distinguishes field states is nonzero on W.

## Why W has no smaller common invariant piece

Complexify W and use additive Fourier characters indexed by nonzero
`k in F_31`.

Translation A has distinct eigenvalues on those 30 character directions, so any
A-invariant complex subspace is the span of some subset of nonzero frequency
directions.

Scaling M sends the k-th character direction to another one with index
proportional to `3k`.  Because 3 is primitive modulo 31, repeated multiplication
by 3 is transitive on all 30 nonzero frequency indices.

Therefore a subset of nonzero frequency directions that is invariant under both
A and M is either:

- empty; or
- all 30 directions.

So W is irreducible for the joint action in this controlled representation.

## Consequence

`ker(L) intersect W` is an invariant subspace.  Because the codes are not all
identical, it cannot equal all of W.  By the argument above it must be zero.
Thus L is injective on W and

\[
\operatorname{rank}(L)\ge \dim W=30.
\]

Hence an exact **linear equivariant** latent representation that distinguishes
the 31 states and supports both generator actions requires

\[
\boxed{d\ge 30}.
\]

## Relation to EXP-022

With only local transition top-1 loss, d<30 can appear successful on each
trained edge while failing the generated grammar.  Isotropic geometry forces the
codebook to use the available rank, but d=28/29 still cannot realize the exact
joint representation.  The experimental jump to exact mixed execution occurs at
d=30 on all tested boundary seeds.

This gives FOG a precise example where latent width is constrained by the
operator representation, not by the number of target classes alone.
