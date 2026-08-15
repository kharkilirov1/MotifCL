# NOTE-015 — Candidate-first JVP sketches

Searching every algebraic candidate with forward-mode AD is unnecessary.  The
scalable protocol is:

1. rank candidate relations using ordinary state residuals;
2. keep a small finalist set;
3. use random JVPs only on the finalists;
4. accept compilation only if both state and tangent residuals pass fixed gates.

For a transition `F`, randomized gains

`||J_F(z) v|| / ||v||`

also provide the production analogue of the local error-gain diagnostic from
NOTE-005 without ever constructing `J_F`.
