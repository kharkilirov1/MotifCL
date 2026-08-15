# IDEA-006 — Operator-compatible latent geometry

Hypothesis: a latent register is useful for reasoning only when important
operators are simple and shared in its coordinate system.

Capacity alone is not enough.  A high-capacity operator can interpolate a
finite transition table in arbitrary coordinates while failing to generalize
or compose when its own output is fed back.

Candidate geometries/operators:

- Fourier / representation-theoretic coordinates for group-like operations;
- learned canonical charts with explicit equivariance losses;
- low-rank tensor-product operator bases;
- typed subspaces where different operator families act on different blocks;
- learned change-of-basis maps constrained by composition identities.

EXP-004 and EXP-005 are the first controlled tests of this idea.
