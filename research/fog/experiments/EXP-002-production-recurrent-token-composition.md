# EXP-002 — Production 10M Recurrent Token Composition

Status: PROTOCOL / implementation phase
Date frozen: 2026-08-14
Depends on: RESULT-001
Target: transfer E3 composition into the real `FOGLatentReasoner` query-bound-v2 path.

## Question

Does recurrent protected binding still compose when we use the real 10M token embeddings, four-layer backbone, reusable K=4 memory, production planner and direct tied vocabulary head?

## Required architecture change

Add a backward-compatible configuration field:

`binding_query_update = "static" | "primary_recurrent"`.

Default must be `static`, so every released v2 checkpoint retains its historical semantics when the field is absent.

For `primary_recurrent`:

- hop 1 binding query = original lexical query code;
- hop t>1 binding query = previous protected primary latent;
- auxiliary workspace may continue to use the contextual backbone query; only the protected binding address is recurrent.

No parameter count change is allowed.

## Task representation

Use a shared identity token for the same state in source, payload and query positions:

`[ROW, STATE_a, VALUE, STATE_f(a), ..., QUERY, STATE_x]`.

This differs intentionally from the old one-hop diagnostic that used unrelated `KEY_*` and `VALUE_*` token IDs. Structural marker tokens and position/offset define role; identity remains the same token code so a payload can become a valid next address.

## Data

Permutation mappings are hash-disjoint across train/validation/test exactly as in the existing structured/token binding gates.

Target at depth R is `f^R(x)`. R is supplied only by latent iteration count.

## Arms

1. production recurrent feedback;
2. production static-query baseline;
3. `R=1` one-hop sanity;
4. corruption after hop 2;
5. optional oracle hard-canonicalized feedback as diagnostic only.

## Initial checkpoint

Start from the released 10M query-bound-v2 token-binding checkpoint. No tuning on locked test.

First run is inference-only. If existing sharpness is insufficient, any further training becomes EXP-003 rather than silently changing this protocol.

## Pass criterion

For the inference-only checkpoint on validation:

- R=1 must retain >=99% exact accuracy;
- recurrent must materially beat static at R=2–4;
- if recurrent reaches >=95% at every R=1–4, extend validation to R=8;
- intermediate corruption must substantially reduce downstream accuracy.

If R=1 fails with shared state tokens, diagnose token-code selection before changing recurrence.

## Implementation smoke (not decisive evidence)

A fresh 10M `query_bound_v2` geometry was run after implementing the opt-in recurrence path. This is a wiring test only because no released checkpoint binary was present in the supplied archive.

On 32 validation examples:

- recurrent R=1..4: **100%, 100%, 100%, 100%**;
- static R=1..4: **100%, 3.125%, 9.375%, 15.625%**;
- recurrent with query corruption after hop 2: **18.75%** at R=3 and R=4;
- recurrent address hit was 100% and the primary aligned with the oracle token code in this fresh-init gate.

Artifact: `artifacts/research/exp_002/smoke.json`.

This establishes that the production code path is wired correctly. It does **not** satisfy EXP-002, which still requires the released trained 10M v2 checkpoint.
