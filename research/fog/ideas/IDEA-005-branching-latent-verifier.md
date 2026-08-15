# IDEA-005 — Branching Workspace and Latent Verifier

Status: IDEA / deliberately deferred
Depends on: stable recurrent composition + typed registers
Evidence: E0 hypothesis

## Goal

Allow multiple latent trajectories to coexist, then score/select/reject them before lexical decoding.

## Why deferred

A verifier cannot be meaningfully evaluated while the base transition itself is not known to compose. Adding a critic now could merely learn shortcuts or compensate for an unstable writer.

## Eventual design questions

- Are branches independent registers or low-rank perturbations of one state?
- Does the verifier score local transition validity or final answer consistency?
- Can a rejected branch roll back to an earlier checkpointed state?
- What is the compute accounting versus simply increasing recurrent depth?
