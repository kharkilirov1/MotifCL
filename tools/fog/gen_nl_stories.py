#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate synthetic English state-tracking stories for the FOG v3 NL-binder.

Each story has a fixed grammar so that token positions of each "action sentence"
end at a known offset, but the *words* vary freely:
    init sentence : "{hero} was in the {LOC}."
    step  sentence: "{hero} {VERB} to the {LOC}."         (R times)

Labels:
    init_state   = index of the LOC in the init sentence
    deltas[k]    = (loc_{k+1} - loc_k) mod 16

The state codebook installed in the FOG model uses tokens 4..19; locations here
are the 16 names mapped 0..15. We emit a compact text representation so the C++
trainer can tokenize and compute deterministic per-sentence positions.

Output: JSONL  {"text": "...", "init": int, "deltas": [int,...]}
"""
import json, random, sys
from pathlib import Path

LOCS = [
    "Kitchen", "Garden", "River", "Forest", "Treehouse", "Lake", "Cave",
    "Bridge", "Hill", "Market", "Barn", "Field", "Orchard", "Well", "Tower",
    "Cellar",
]

HEROES = ["Alice", "Lily", "Tom", "Mia", "Sam", "A brave knight", "A little robot",
          "The fairy queen", "A wise owl", "Robin", "A kind witch", "A tiny mouse"]

VERBS = ["went", "ran", "walked", "skipped", "hurried", "journeyed",
         "climbed up", "wandered", "dashed"]

N_TRAIN = 6000
N_TEST  = 1000
SEED_T = 12345
SEED_E = 20260816
MAX_R = 4      # curricula R=1..4
MIN_R = 1


def make_story(rng):
    hero = rng.choice(HEROES)
    n_states = len(LOCS)
    R = rng.randint(MIN_R, MAX_R)
    states = [rng.randrange(n_states)]
    for _ in range(R):
        # bias toward non-trivial but legal moves
        d = rng.randrange(0, n_states)           # 0..15 inclusive
        states.append((states[-1] + d) % n_states)
    deltas = [(states[k + 1] - states[k]) % n_states for k in range(R)]
    parts = [f"{hero} was in the {LOCS[states[0]]}."]
    for k in range(R):
        parts.append(f"{hero} {rng.choice(VERBS)} to the {LOCS[states[k+1]]}.")
    text = " ".join(parts)
    return {"text": text, "init": states[0], "deltas": deltas, "R": R,
            "states": states}


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    for name, n, seed in [("fog_nl_train.jsonl", N_TRAIN, SEED_T),
                          ("fog_nl_test.jsonl",  N_TEST,  SEED_E)]:
        rng = random.Random(seed)
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(json.dumps(make_story(rng), ensure_ascii=False) + "\n")
        print(f"wrote {n} stories -> {path}")


if __name__ == "__main__":
    main()