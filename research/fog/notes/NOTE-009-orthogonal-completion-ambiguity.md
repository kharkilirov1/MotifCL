# NOTE-009 — k transition constraints leave an O(d-k) completion ambiguity

Let `T` be an unknown orthogonal operator on `R^d`.  Suppose its action is known
on `k` linearly independent input vectors and the corresponding images are
consistent with orthogonality.

The known inputs span a k-dimensional subspace U.  T is fixed on U.  On the
orthogonal complement `U^perp`, any orthogonal map can be composed with T
without changing any observed correspondence.

Therefore the residual ambiguity is

\[
O(d-k).
\]

Important boundaries:

- `k=d-2`: residual `O(2)` is continuous;
- `k=d-1`: residual `O(1)={+1,-1}` is one orientation bit;
- `k=d`: no unconstrained orthogonal complement remains.

EXP-027/028 display exactly this pattern for d=30.  The bad k=29 MUL seed has
`det(M)~+1` while successful completions have `det(M)~-1`.  A structural motif
constraint can resolve the last discrete bit, but cannot generally infer a full
continuous O(2) completion from the same evidence.

This yields a sample-complexity principle for FOG: operator-family structure can
reduce transition supervision by the degrees of freedom that the family itself
removes.
