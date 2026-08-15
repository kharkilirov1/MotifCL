# EXP-020 — Representation width vs operator-orbit size

Status: **PASSED CONTROLLED**  
Date: 2026-08-14

For multiplier subgroups `H <= F_31^*` of sizes

`1,2,3,5,6,10,15,30`,

choose the additive-character frequency set to be one orbit `S=H`.

Then:

- ADD remains local in every retained frequency;
- MUL by any `y in H` permutes `S` into itself;
- real representation width is `2|H|`.

Evaluate random mixed ADD / MUL-by-H programs through depth 128.
