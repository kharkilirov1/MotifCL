# IDEA-009 — Structural Operator Compiler

A learned dense latent transition may be simple only after a change of gauge.
Treat training and execution as two layers:

1. neural learner estimates a recurrent representation and approximate actions;
2. structural compiler discovers finite-order/spectral/operator relations,
   canonicalizes the gauge, extracts sparse motifs, removes redundant operators
   and projects the result back onto a recurrently legal family.

Primary questions:

- can the compiler infer laws without task-semantic metadata?
- does motif projection improve long-horizon stability under learned noise?
- can independently trained modules be aligned through invariants rather than a
  learned paired-state bridge?
- how does the number of observed transitions interact with operator-family
  constraints?

Controlled evidence: EXP-023 through EXP-029.
