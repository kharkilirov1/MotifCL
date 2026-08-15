# IDEA-003 — Typed Latent Registers Without Identity Destruction

Status: IDEA
Depends on: EXP-001
Evidence: E0 hypothesis

## Motivation

A recurrent machine benefits from explicit roles (`address`, `value`, `control`, `scratch`), but role encoding must not destroy the semantic identity required for reuse.

Bad representation:

`KEY_B` and `VALUE_B` are unrelated codes.

Then `VALUE_B` cannot naturally address the row keyed by `KEY_B`.

Preferred factorization:

`state = identity(B) +/x role(value)`

with a binding comparator that can read `identity(B)` independently of role.

## Candidate designs

- Separate subspaces: `[identity | role | payload-extra]`.
- Multiplicative/FiLM role modulation with an invariant identity projection.
- Dedicated register type outside the payload vector; binder receives type through routing metadata.
- Learned value->address bridge only if it is forced to generalize to unseen identities.

## Strong test

Hold out some state identities entirely from bridge training. If a role bridge works only for seen identities, it is a codebook translation table, not a reusable type conversion.
