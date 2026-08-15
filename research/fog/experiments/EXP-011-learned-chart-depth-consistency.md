# EXP-011 — Multi-depth consistency for learned-chart stability

Status: **PASSED**  
Date: 2026-08-14

EXP-010 showed 100% binary generalization but seed-sensitive long recurrence.

The chart is trained on the *same successor law*, but terminal latent consistency
is required after repeated `+1` programs of depths `{1,2,3}`.

No arbitrary `(a,b)->a+b` target pairs are exposed.  This is a composition
constraint on one shared transition, not extra table supervision.

Evaluation:

- all held-out binary pairs;
- recurrent random-operand programs at depths 2,4,8,16,32,64;
- three seeds.
