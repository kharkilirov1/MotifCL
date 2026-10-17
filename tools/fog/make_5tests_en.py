#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate the 5 user state-tracking tests into English narratives using the
NL-binder's fixed 16-location vocabulary, and pretokenize to .tok format:
    <init> <deltas> <token_ids>

Locations (indices 0..15) reused metaphorically for the logic puzzles:
  Kitchen Garden River Forest Treehouse Lake Cave
  Bridge Hill Market Barn Field Orchard Well Tower Cellar
"""
import json
from pathlib import Path
from tokenizers import Tokenizer

TOK = Tokenizer.from_file("ports/rx580/fog/tinystories_3k_bpe.json")

def loc(i):
    return ["Kitchen","Garden","River","Forest","Treehouse","Lake","Cave",
            "Bridge","Hill","Market","Barn","Field","Orchard","Well","Tower","Cellar"][i]

def story_to_tok(story):
    ids = TOK.encode(story).ids
    return ids

# Each puzzle represented as a visit-sequence; we synthesize its narrative + deltas.
VERBS = ["went to","walked to","ran to","skipped to","hurried to",
         "journeyed to","climbed up to","wandered to","dashed to"]

# --- TEST 1 : Alice & silver key, 10-step long chain (locations = key-holder moves) ---
# Key visits: Kitchen -> pocket/Garden -> (still pocket)River -> (still)Forest ->
#  Box(Cave) -> (still)Bridge -> Treehouse-shelf(Hill) -> (stays)->(stays). We map
#  to math the binder learned: every sentence moves the *key holder*.
t1_states = [0,1,2,3,6,7,8,9,10,11,13]      # 10 moves
# deltas: 1,1,1,3,1,1,1,1,1,2

# --- TEST 2 : 4 objects, run as 4 sub-stories ---
# Sword (Knight->Chest->Knight): Market->Barn->Market
t2a_states = [9,10,9]
# Ring  (Princess->Box in Room -> stays): Garden->Cave->Cave
t2b_states = [1,6,6]
# Coin  (Dragon holds -> still holds): Orchard stays
t2c_states = [12,12]
# Book  (Wizard -> Shelf -> Chest): Barn->Tower->Cellar
t2d_states = [10,14,15]

# --- TEST 3 : nested conditions + fake premise, 4 coins -> Shield ---
# We map the coin-count reading onto the state index and one FOG rule-step:
#   state S4 == "4 coins", delta=(Shield_idx - 4) mod 16 to land on S14 (Tower == Shield)
# Story wording pretends the locations are the puzzle's entities.
t3_states = [4,14]                          # 1 move, delta=10

# --- TEST 4 : rule shift + long chain : S0 +1+1+1 then +3+3+3+3 = S15 ---
t4_states = [0,1,2,3,6,9,12,15]            # 7 moves, deltas 1,1,1,3,3,3,3

# --- TEST 5 : same as T4 (its on/off split happens in the binary) ---
# (we reuse t4 for the fog-on/off contrast; no separate story needed here)

def render_hero(hero, states):
    parts = [f"{hero} was in the {loc(states[0])}."]
    for k in range(1, len(states)):
        parts.append(f"{hero} {VERBS[k % len(VERBS)]} the {loc(states[k])}.")
    return " ".join(parts)

def deltas_of(states):
    out=[]
    for k in range(1,len(states)):
        out.append((states[k]-states[k-1]) % 16)
    return out

tests = [
    ("TEST 1: Long chain (10 steps) + distractors (Alice & the silver key)", "Alice", t1_states),
    ("TEST 2a: Sword (Knight <-> Chest)",                  "Thomas",  t2a_states),
    ("TEST 2b: Ring  (Princess -> Box in Room, stays)",    "Lily",    t2b_states),
    ("TEST 2c: Gold coin (Dragon holds -> still holds)",  "Sam",     t2c_states),
    ("TEST 2d: Book   (Wizard -> Shelf -> Chest)",         "Mia",     t2d_states),
    ("TEST 3: Nested conditions + fake premise (4 coins -> Shield)", "The fairy queen", t3_states),
    ("TEST 4: Rule shift + long chain (S0 +1x3 then +3x4 = S15)", "Tom", t4_states),
    ("TEST 5: Fog on/off diagnostic (reuses T4)",         "Tom",     t4_states),
]

lines = []
desc = []
for title, hero, states in tests:
    story = render_hero(hero, states)
    d = deltas_of(states)
    ids = story_to_tok(story)
    dcsv = ",".join(str(x) for x in d)
    tcsv = ",".join(str(x) for x in ids)
    lines.append(f"{states[0]} {dcsv} {tcsv}")
    desc.append(f"# {title}\n# init=S{states[0]} ({loc(states[0])})  deltas={d}  gold=S{states[-1]} ({loc(states[-1])})\n# story: {story}")

p = Path("data/fog_nl_5tests.tok")
with p.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
pd = Path("data/fog_nl_5tests.descriptions.txt")
with pd.open("w", encoding="utf-8") as f:
    f.write("\n\n".join(desc) + "\n")

print("wrote", p, "with", len(lines), "tests")
print(pd, "descriptions:")
print("\n\n".join(desc))