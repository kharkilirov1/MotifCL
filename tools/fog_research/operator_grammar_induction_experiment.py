#!/usr/bin/env python3
"""EXP-015: infer an operator from demonstrations using a finite latent grammar.

Two cyclic operator modules are learned from local successor laws only:

- ADD over F_31, represented as the cyclic additive group of order 31;
- MUL over F_31^*, represented in primitive-root exponent coordinates, a cyclic
  group of order 30.

At evaluation there is *no operation label*.  An episode supplies K input/output
demonstrations plus one query pair.  Each candidate operator predicts the demo
outputs in its own learned latent chart.  A compare/select router scores the
candidate by mean latent cosine consistency and selects the best operator for
answering the query.

This is a controlled operator-induction gate: the candidate grammar is finite
and hand-specified, while the charts/coordinate frames are learned.  It tests
whether operator selection can be separated from operator execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENT_NAME = "exp_015_operator_grammar_induction"
PRIME = 31
PRIMITIVE_ROOT = 3


def discrete_log_table(p: int = PRIME, g: int = PRIMITIVE_ROOT) -> tuple[list[int], list[int]]:
    exp_to_value = [pow(g, e, p) for e in range(p - 1)]
    value_to_exp = [-1] * p
    for e, v in enumerate(exp_to_value):
        value_to_exp[v] = e
    return exp_to_value, value_to_exp


@dataclass(frozen=True)
class Config:
    harmonics: int = 4
    head_scale: float = 20.0
    separation_weight: float = 0.05


class LearnedCyclicOperator(nn.Module):
    """Cyclic group chart with a normed law-by-construction operator."""

    def __init__(self, order: int, cfg: Config, seed: int):
        super().__init__()
        self.order = order
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.phase = nn.Parameter(
            torch.rand(order, cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )
        self.frame_angle = nn.Parameter(
            torch.rand(cfg.harmonics, generator=g) * 2 * math.pi - math.pi
        )

    def codebook(self) -> torch.Tensor:
        phase = self.phase - self.phase[:1]
        return F.normalize(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1), dim=-1)

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.cfg.harmonics
        ar, ai = a[..., :h], a[..., h:]
        br, bi = b[..., :h], b[..., h:]
        an = torch.sqrt(ar.square() + ai.square()).clamp_min(1e-8)
        bn = torch.sqrt(br.square() + bi.square()).clamp_min(1e-8)
        ar, ai, br, bi = ar / an, ai / an, br / bn, bi / bn
        c, s = torch.cos(self.frame_angle), torch.sin(self.frame_angle)
        a_r, a_i = c * ar - s * ai, s * ar + c * ai
        b_r, b_i = c * br - s * bi, s * br + c * bi
        p_r = a_r * b_r - a_i * b_i
        p_i = a_r * b_i + a_i * b_r
        out_r, out_i = c * p_r + s * p_i, -s * p_r + c * p_i
        return F.normalize(torch.cat((out_r, out_i), dim=-1), dim=-1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cfg.head_scale * F.normalize(z, dim=-1) @ self.codebook().T


def train_cyclic(model: LearnedCyclicOperator, *, seed: int, steps: int, lr: float) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ids = torch.arange(model.order)
    for _ in range(steps):
        code = model.codebook()
        one = code[torch.ones_like(ids)]
        losses = []
        for depth in (1, 2, 3):
            state = code[ids]
            for _ in range(depth):
                state = model.op(state, one)
            target = code[(ids + depth) % model.order]
            losses.append((1 - F.cosine_similarity(state, target, dim=-1)).mean())
        sep = F.cross_entropy(model.cfg.head_scale * code @ code.T, ids)
        loss = torch.stack(losses).mean() + model.cfg.separation_weight * sep
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()


@torch.inference_mode()
def module_accuracy(model: LearnedCyclicOperator) -> float:
    ids = torch.arange(model.order)
    a = ids.repeat_interleave(model.order)
    b = ids.repeat(model.order)
    target = (a + b) % model.order
    pred = model.logits(model.op(model.codebook()[a], model.codebook()[b])).argmax(-1)
    return float(pred.eq(target).float().mean())


def sample_nonzero_add_pair(g: torch.Generator) -> tuple[int, int, int]:
    while True:
        a = int(torch.randint(1, PRIME, (1,), generator=g))
        b = int(torch.randint(1, PRIME, (1,), generator=g))
        c = (a + b) % PRIME
        if c != 0:
            return a, b, c


def sample_mul_pair(g: torch.Generator) -> tuple[int, int, int]:
    a = int(torch.randint(1, PRIME, (1,), generator=g))
    b = int(torch.randint(1, PRIME, (1,), generator=g))
    return a, b, (a * b) % PRIME


@torch.inference_mode()
def candidate_demo_score(
    name: str,
    model: LearnedCyclicOperator,
    demos: list[tuple[int, int, int]],
    value_to_exp: list[int],
) -> float:
    code = model.codebook()
    scores = []
    for a, b, c in demos:
        if name == "add":
            ia, ib, ic = a, b, c
        else:
            if min(a, b, c) <= 0:
                return float("-inf")
            ia, ib, ic = value_to_exp[a], value_to_exp[b], value_to_exp[c]
        pred = model.op(code[ia : ia + 1], code[ib : ib + 1])[0]
        scores.append(float(F.cosine_similarity(pred[None], code[ic : ic + 1], dim=-1)[0]))
    return sum(scores) / max(1, len(scores))


@torch.inference_mode()
def answer_query(
    name: str,
    model: LearnedCyclicOperator,
    query: tuple[int, int],
    exp_to_value: list[int],
    value_to_exp: list[int],
) -> int:
    a, b = query
    code = model.codebook()
    if name == "add":
        idx = int(model.logits(model.op(code[a : a + 1], code[b : b + 1])).argmax(-1)[0])
        return idx
    ia, ib = value_to_exp[a], value_to_exp[b]
    e = int(model.logits(model.op(code[ia : ia + 1], code[ib : ib + 1])).argmax(-1)[0])
    return exp_to_value[e]


@torch.inference_mode()
def evaluate_episodes(
    add_model: LearnedCyclicOperator,
    mul_model: LearnedCyclicOperator,
    *,
    seed: int,
    demonstrations: int,
    episodes: int,
) -> dict:
    exp_to_value, value_to_exp = discrete_log_table()
    g = torch.Generator().manual_seed(seed + 40000 + demonstrations)
    route_correct = 0
    answer_correct = 0
    score_ambiguous = 0
    semantic_ambiguous = 0
    nonambig_route = 0
    nonambig_answer = 0
    nonambig_total = 0
    margins = []

    for _ in range(episodes):
        true_name = "add" if int(torch.randint(2, (1,), generator=g)) == 0 else "mul"
        sampler = sample_nonzero_add_pair if true_name == "add" else sample_mul_pair
        demos = [sampler(g) for _ in range(demonstrations)]
        qa, qb, qc = sampler(g)
        add_score = candidate_demo_score("add", add_model, demos, value_to_exp)
        mul_score = candidate_demo_score("mul", mul_model, demos, value_to_exp)
        scores = {"add": add_score, "mul": mul_score}
        predicted_name = max(scores, key=scores.get)
        is_score_ambiguous = abs(add_score - mul_score) < 1e-5
        score_ambiguous += int(is_score_ambiguous)
        is_semantic_ambiguous = all(
            ((a + b) % PRIME == c) and ((a * b) % PRIME == c)
            for a, b, c in demos
        )
        semantic_ambiguous += int(is_semantic_ambiguous)
        route_correct += int(predicted_name == true_name)
        predicted_value = answer_query(predicted_name, add_model if predicted_name == "add" else mul_model, (qa, qb), exp_to_value, value_to_exp)
        answer_correct += int(predicted_value == qc)
        if not is_semantic_ambiguous:
            nonambig_total += 1
            nonambig_route += int(predicted_name == true_name)
            nonambig_answer += int(predicted_value == qc)
        margins.append(abs(add_score - mul_score))

    return {
        "demonstrations": demonstrations,
        "episodes": episodes,
        "route_accuracy": route_correct / episodes,
        "answer_accuracy": answer_correct / episodes,
        "score_ambiguous_fraction": score_ambiguous / episodes,
        "semantic_ambiguous_fraction": semantic_ambiguous / episodes,
        "nonambiguous_route_accuracy": nonambig_route / max(1, nonambig_total),
        "nonambiguous_answer_accuracy": nonambig_answer / max(1, nonambig_total),
        "mean_score_margin": sum(margins) / len(margins),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_015/metrics.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[30, 31, 32])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config()
    rows = []
    for seed in args.seeds:
        add_model = LearnedCyclicOperator(31, cfg, seed)
        mul_model = LearnedCyclicOperator(30, cfg, seed + 1000)
        train_cyclic(add_model, seed=seed, steps=args.steps, lr=args.lr)
        train_cyclic(mul_model, seed=seed + 1000, steps=args.steps, lr=args.lr)
        rows.append({
            "seed": seed,
            "add_full_group_accuracy": module_accuracy(add_model),
            "mul_exponent_group_accuracy": module_accuracy(mul_model),
            "episode_metrics": [evaluate_episodes(add_model, mul_model, seed=seed, demonstrations=k, episodes=args.episodes) for k in (1, 2, 3)],
        })
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": asdict(cfg),
        "candidate_operators": ["ADD_mod_31", "MUL_nonzero_mod_31"],
        "explicit_operation_label_at_evaluation": False,
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
