# EXP-015 — Operator induction from demonstrations

Status: **ROUTER PASSED; FREE EXECUTOR PARTIAL**  
Date: 2026-08-14

Two learned cyclic operator candidates represent ADD over `F_31` and MUL over
`F_31^*`.  At evaluation no operation label is supplied.  Each candidate scores
K demonstrations by latent output consistency; compare/select chooses the
operator for a new query.

Semantic ambiguity is measured explicitly: with one demonstration, some triples
satisfy both ADD and MUL and no router can distinguish them from that evidence.
