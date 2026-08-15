#!/usr/bin/env python3
"""Exact multi-digit binding gate for the protected FOG v2 payload channel.

Each unseen table maps eight symbolic keys to independent fixed-length digit
strings.  A single differentiable address distribution selects a whole row;
the selected payload is represented by one protected latent per digit.  Digit
embeddings and the vocabulary classifier are the same frozen orthogonal
codebook, so success requires coordinate-preserving digit storage rather than
an adaptable decoder learning an arbitrary latent code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from fog_lmw.checkpoint import atomic_torch_save
from fog_lmw.core import RMSNorm
from fog_lmw.motifs import AddressedPayloadBinding
from matched_structured_lookup_experiment import (
    StructuredTaskConfig,
    initialize_by_parameter_name,
    target_deranged_indices,
)


EXPERIMENT_NAME = "fog_exact_multidigit_binding_v1"
Split = Literal["train", "validation", "test"]
Intervention = Literal["normal", "zero", "target_deranged", "query_deranged"]


@dataclass(frozen=True)
class DigitTaskConfig:
    table_size: int = 8
    digits: int = 4
    radix: int = 10

    def validate(self) -> None:
        if self.table_size < 2 or self.digits < 2 or self.radix < 2:
            raise ValueError("table_size, digits, and radix must be >=2")


@dataclass(frozen=True)
class DigitBatch:
    row_sources: torch.Tensor
    row_digits: torch.Tensor
    query_keys: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> "DigitBatch":
        return DigitBatch(
            self.row_sources.to(device),
            self.row_digits.to(device),
            self.query_keys.to(device),
            self.targets.to(device),
        )


def _stable_seed(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _bucket(table: tuple[tuple[int, ...], ...]) -> int:
    raw = bytes(digit for row in table for digit in row)
    return hashlib.blake2b(raw, digest_size=8, person=b"fogdigv1").digest()[0] % 10


def _wanted_bucket(split: Split) -> set[int]:
    return set(range(8)) if split == "train" else ({8} if split == "validation" else {9})


def make_digit_batch(
    task: DigitTaskConfig,
    *,
    data_seed: int,
    split: Split,
    start_index: int,
    batch_size: int,
) -> DigitBatch:
    task.validate()
    rows = []
    for sample_index in range(start_index, start_index + batch_size):
        attempt = 0
        while True:
            rng = random.Random(_stable_seed(EXPERIMENT_NAME, data_seed, sample_index, attempt))
            table = tuple(
                tuple(rng.randrange(task.radix) for _ in range(task.digits))
                for _ in range(task.table_size)
            )
            if _bucket(table) in _wanted_bucket(split):
                break
            attempt += 1
        query = rng.randrange(task.table_size)
        order = list(range(task.table_size))
        rng.shuffle(order)
        rows.append((order, [table[source] for source in order], query, table[query]))
    return DigitBatch(
        row_sources=torch.tensor([row[0] for row in rows], dtype=torch.long),
        row_digits=torch.tensor([row[1] for row in rows], dtype=torch.long),
        query_keys=torch.tensor([row[2] for row in rows], dtype=torch.long),
        targets=torch.tensor([row[3] for row in rows], dtype=torch.long),
    )


class ExactDigitBinder(nn.Module):
    def __init__(self, task: DigitTaskConfig, d_model: int, compare_rank: int):
        super().__init__()
        task.validate()
        if d_model < task.table_size + task.radix:
            raise ValueError("d_model must fit disjoint key and digit codebooks")
        self.task = task
        self.d_model = d_model
        self.key = nn.Embedding(task.table_size, d_model)
        self.digit = nn.Embedding(task.radix, d_model)
        self.row_type = nn.Parameter(torch.empty(1, 1, d_model))
        self.query_type = nn.Parameter(torch.empty(1, d_model))
        self.address_norm = RMSNorm(d_model)
        self.bind = AddressedPayloadBinding(d_model, compare_rank)
        self.payload_norm = RMSNorm(d_model)

    def reason(
        self,
        batch: DigitBatch,
        *,
        query_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        addresses = self.key(batch.row_sources) + self.row_type
        payloads = self.digit(batch.row_digits)  # [B, rows, digits, D]
        query = self.key(batch.query_keys) + self.query_type
        if query_override is not None:
            query = query_override
        # AddressedPayloadBinding selects rank-3 payloads, so flatten the digit
        # axis into D only for the weighted copy, then restore it unchanged.
        batch_size, row_count, digit_count, width = payloads.shape
        flattened = payloads.reshape(batch_size, row_count, digit_count * width)
        # Reuse only Q/K projections from the canonical binding motif because
        # its payload API intentionally requires the same width as addresses.
        q = self.bind.q_proj(query[:, None])
        k = self.bind.k_proj(self.address_norm(addresses))
        score = torch.matmul(q, k.transpose(-1, -2)).squeeze(1)
        score = score / (self.bind.compare_rank ** 0.5)
        attention = score.softmax(dim=-1)
        selected = torch.einsum("br,brp->bp", attention, flattened)
        selected = selected.reshape(batch_size, digit_count, width)
        return self.payload_norm(selected), attention

    def logits(
        self,
        batch: DigitBatch,
        intervention: Intervention = "normal",
    ) -> torch.Tensor:
        query_override = None
        if intervention == "query_deranged":
            counterfactual_keys = (batch.query_keys + 1) % self.task.table_size
            query_override = self.key(counterfactual_keys) + self.query_type
        payload, _ = self.reason(batch, query_override=query_override)
        if intervention == "zero":
            payload = torch.zeros_like(payload)
        elif intervention == "target_deranged":
            # Derange by the complete digit string, not only one digit.
            codes = torch.tensor(
                [int("".join(map(str, row.tolist()))) for row in batch.targets.cpu()],
                device=batch.targets.device,
            )
            payload = payload.index_select(0, target_deranged_indices(codes))
        elif intervention not in ("normal", "query_deranged"):
            raise ValueError(intervention)
        # Frozen tied digit codebook: no learned reader can hide corruption.
        return torch.einsum("bld,vd->blv", payload, self.digit.weight)


@torch.no_grad()
def build_model(task: DigitTaskConfig, *, d_model: int, rank: int, seed: int) -> ExactDigitBinder:
    model = ExactDigitBinder(task, d_model, rank)
    initialize_by_parameter_name(model, seed=seed, std=0.02)
    model.key.weight.zero_()
    model.key.weight[:, : task.table_size].copy_(torch.eye(task.table_size))
    model.key.weight.requires_grad_(False)
    model.digit.weight.zero_()
    start = task.table_size
    model.digit.weight[:, start : start + task.radix].copy_(torch.eye(task.radix))
    model.digit.weight.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate(
    model: ExactDigitBinder,
    task: DigitTaskConfig,
    *,
    data_seed: int,
    split: Split,
    examples: int,
    batch_size: int,
    device: torch.device,
    intervention: Intervention,
) -> dict:
    model.eval()
    exact = 0
    digit_correct = 0
    nll = 0.0
    for start in range(0, examples, batch_size):
        count = min(batch_size, examples - start)
        batch = make_digit_batch(
            task, data_seed=data_seed, split=split, start_index=start, batch_size=count
        ).to(device)
        logits = model.logits(batch, intervention)
        prediction = logits.argmax(dim=-1)
        exact += int(prediction.eq(batch.targets).all(dim=-1).sum())
        digit_correct += int(prediction.eq(batch.targets).sum())
        nll += float(F.cross_entropy(
            logits.reshape(-1, task.radix), batch.targets.reshape(-1), reduction="sum"
        ))
    return {
        "split": split,
        "intervention": intervention,
        "examples": examples,
        "exact": exact,
        "exact_match": exact / examples,
        "digit_accuracy": digit_correct / (examples * task.digits),
        "nll_per_digit": nll / (examples * task.digits),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def train_seed(args: argparse.Namespace, task: DigitTaskConfig, seed: int) -> dict:
    device = torch.device(args.device)
    model = build_model(task, d_model=args.d_model, rank=args.compare_rank, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    trace = []
    for step in range(args.steps):
        batch = make_digit_batch(
            task,
            data_seed=args.data_seed,
            split="train",
            start_index=step * args.batch_size,
            batch_size=args.batch_size,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(batch)
        loss = F.cross_entropy(logits.reshape(-1, task.radix), batch.targets.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            exact = logits.argmax(-1).eq(batch.targets).all(-1).float().mean()
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "batch_exact": float(exact.detach()),
                "grad_norm": float(grad_norm),
            }
            trace.append(row)
            print(f"[digits seed={seed}] {step+1}/{args.steps} loss={row['loss']:.4f} exact={100*row['batch_exact']:.1f}%", flush=True)
    evaluations = {
        mode: evaluate(
            model,
            task,
            data_seed=args.data_seed,
            split="validation",
            examples=args.eval_examples,
            batch_size=args.eval_batch_size,
            device=device,
            intervention=mode,
        )
        for mode in ("normal", "zero", "target_deranged", "query_deranged")
    }
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "seed": seed,
        "task": asdict(task),
        "steps": args.steps,
        "trace": trace,
        "eval": evaluations,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    directory = args.output_dir / f"seed_{seed:04d}"
    atomic_torch_save(
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "seed": seed,
            "task": asdict(task),
            "d_model": args.d_model,
            "compare_rank": args.compare_rank,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metrics": metrics,
        },
        directory / "model.pt",
    )
    _write_json(directory / "metrics.json", metrics)
    return metrics


def checkpoint_eval(args: argparse.Namespace) -> None:
    payload = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
    task = DigitTaskConfig(**payload["task"])
    model = build_model(task, d_model=payload["d_model"], rank=payload["compare_rank"], seed=payload["seed"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(args.device))
    result = {
        "experiment": EXPERIMENT_NAME,
        "split": args.evaluation_split,
        "eval": {
            mode: evaluate(
                model, task, data_seed=args.data_seed, split=args.evaluation_split,
                examples=args.eval_examples, batch_size=args.eval_batch_size,
                device=torch.device(args.device), intervention=mode,
            )
            for mode in ("normal", "zero", "target_deranged", "query_deranged")
        },
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluate-checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/binding_digits"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-seed", type=int, default=20260813)
    p.add_argument("--table-size", type=int, default=8)
    p.add_argument("--digits", type=int, default=4)
    p.add_argument("--radix", type=int, default=10)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--compare-rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--eval-examples", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    if args.evaluate_checkpoint:
        if args.output is None:
            raise ValueError("--output is required")
        checkpoint_eval(args)
        return
    if args.evaluation_split == "test":
        raise ValueError("training is validation-only; test saved checkpoints separately")
    task = DigitTaskConfig(args.table_size, args.digits, args.radix)
    runs = [train_seed(args, task, seed) for seed in args.seeds]
    summary = {"experiment": EXPERIMENT_NAME, "test_split_touched": False, "runs": runs}
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"validation_exact": [r["eval"]["normal"]["exact_match"] for r in runs]}, indent=2))


if __name__ == "__main__":
    main()
