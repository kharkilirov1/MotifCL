# EXP-037 — Model-side generated value + recurrent re-addressing

## Question

Does the new `register_machine_v3` architecture do more than differentiate and
route READ?  Can a full tiny FOG model generate a value absent from the payload
memory and reuse that continuous value as the next address?

## Task

Each example contains a random table `state -> operand`.  Starting from query
state `s`, every latent tick must execute

` s <- s + table[s] (mod 8) `.

The table stores only operands, never the ready next-state answer.  The exact
binding lane reads the operand; the value register must then generate the new
state, and that generated state becomes the next binding address.

Only the machine cell is trained.  The state-token chart and exact binder are
frozen.  Training depths are 1--3; evaluation depths include 4, 6 and 8.

## Architectural ablation discovered during the experiment

A soft mixture over legal operator candidates drifts off the canonical manifold
even when a correct structured primitive is present.  v3 therefore uses
straight-through **hard operator routing** and includes a parameter-free
`BLOCK_PRODUCT` primitive alongside READ, IDENTITY and flexible low-rank
bilinear operators.

## Result

**PASS on 3/3 seeds.**

After 150 steps:

- exact accuracy at every evaluated depth 1,2,3,4,6,8;
- minimum mean target cosine across all seed/depth arms: > 0.9999979;
- routing converges predominantly to `BLOCK_PRODUCT`; READ remains selected only
  on cases where reading the operand is itself semantically correct;
- flexible candidates are not needed by this controlled law.

Artifacts: `artifacts/research/exp_037/`.
