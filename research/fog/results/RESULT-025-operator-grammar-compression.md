# RESULT-025 — A redundant learned operator library compiles to two generators

On new d=30 seeds 80/81/82 the compiler independently selects `S3` as the
primitive scaling generator of order 30 and discovers

\[
S5\approx S3^{20},
\qquad
S7\approx S3^{28}.
\]

The redundant dense `S5` and `S7` matrices can therefore be removed.  Random
programs executed through the minimal library `{A,S3}` remain **100% at depth
64** on all three seeds.

On matched d=29 runs, the discovered power residuals for S5/S7 are roughly
`0.81–0.96`; the compiler rejects the grammar on all three seeds.

Interpretation: operator induction can include **relation discovery and library
minimization**, not just selection among a fixed candidate set.

Evidence: `artifacts/research/exp_026/metrics.json`.
