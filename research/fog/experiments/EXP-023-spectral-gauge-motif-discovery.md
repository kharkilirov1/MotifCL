# EXP-023 — Spectral gauge motif discovery

Status: **PASSED**  
Date: 2026-08-14

Starting point: the learned dense `d=30` invariant representation from EXP-022.
No Fourier basis is supplied.

Protocol:

1. diagonalize learned `A` only;
2. express learned `M` in that eigengauge;
3. hard-prune `A` to its diagonal and `M` to one entry per input eigenspace;
4. transform the pruned actions back to the original real gauge;
5. evaluate recurrent mixed programs at depth 64.

Matched control: `d=29` EXP-022 representation.

Success criterion: a sparse action is accepted only if pruning preserves the
long recurrent program, not merely the one-step edges.
