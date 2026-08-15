# EXP-018 — Infer a primitive, then execute it recurrently

Status: **PASSED CONTROLLED**  
Date: 2026-08-14

Two non-ambiguous demonstrations identify a hidden homogeneous operator (ADD or
MUL).  The selected learned EXP-016 module then receives a start value and a
sequence of operands.  Its continuous output is reused directly at every hop;
there is no intermediate decode/snap.

Mixed ADD/MUL instruction sequences are intentionally excluded here so chart
switching is not hidden inside the result.
