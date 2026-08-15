# IDEA-008 — Invariant latent register + finite operator grammar

## Core idea

Represent the reusable machine state in a space that is **closed under the
operator family**, then let different operator motifs act on that same register.

The operator does not have to look like the representation.  In EXP-019 the
same additive-character register supports:

- ADD as local per-frequency complex multiplication;
- MUL as operand-conditioned frequency permutation.

## Why this is different from a generic Transformer block

A generic block must discover both the law and the coordinates from loss.  The
FOG alternative is a grammar of low-complexity action motifs, for example:

- compare/select/address;
- local product / affine update;
- permutation/group action;
- accumulation;
- canonical/attractor projection;
- control/PC/HALT;
- sparse chart CAST when no shared invariant basis is economical.

A router can infer/select the motif from demonstrations or control state, as in
EXP-017/018.

## Research question

Can these action motifs and their invariant subspaces be learned from ordinary
data, while retaining structural recurrence guarantees?

## Falsifiers

- learned basis grows to a dense table with no OOD operator/depth transfer;
- sparse action disappears when identities/operators are held out;
- routing succeeds only because an explicit task label leaks the operator;
- recurrent closure/gain diagnostics predict instability despite one-step fit.
