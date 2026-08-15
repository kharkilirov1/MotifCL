# EXP-026 — Automatic grammar compression

Status: **PASSED d=30 / rejected d=29 control**  
Date: 2026-08-14

Learn one shared representation and four dense actions independently from
transition tables:

- `A: x->x+1`;
- `S3: x->3x`;
- `S5: x->5x`;
- `S7: x->7x`.

No action relations are supplied during training.  Afterwards, the compiler
searches finite orders and direct matrix-power relations, selects the
highest-order scaling action as a primitive, deletes redundant scaling matrices
and executes programs with only the minimal generator library.
