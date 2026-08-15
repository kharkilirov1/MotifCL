#!/usr/bin/env python3
"""EXP-001: recurrent protected-binding composition without tokenized intermediates.

This controlled gate isolates the next unresolved question after binding-v2:
can the selected payload at hop t become the address at hop t+1?

Each example is an unseen permutation table f over N states and a start state x.
The target at depth R is f^R(x).  R is never encoded in the input; it exists only
as the number of recurrent latent transitions.

The minimal gate deliberately uses one canonical frozen state codebook for both
row sources and payloads.  That separates semantic identity from role and avoids
confounding composition with a KEY-code -> VALUE-code namespace translation.
Only the address softmax sharpness is trainable.

Validation is the default.  The locked test split must be evaluated only from a
saved checkpoint after the protocol pass criteria have been frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from fog_lmw.checkpoint import atomic_torch_save
from fog_lmw.core import RMSNorm
from fog_lmw.motifs import AddressedPayloadBinding
from fog_lmw.planner import CosineTiedHead
from matched_structured_lookup_experiment import (
    StructuredBatch,
    StructuredTaskConfig,
    _update_stream_digest,
    make_batch,
    verify_mapping_holdout,
)


EXPERIMENT_NAME = "exp_001_recurrent_binding_composition"
Mode = Literal["recurrent", "static", "hard_recurrent"]
Intervention = Literal["normal", "corrupt_after_2"]


@dataclass(frozen=True)
class CompositionConfig:
    d_model: int = 32
    initial_scale: float = 1.5

    def validate(self, task: StructuredTaskConfig) -> None:
        if self.d_model < task.table_size:
            raise ValueError("d_model must be >= table_size for orthogonal state codes")
        if self.initial_scale <= 0:
            raise ValueError("initial_scale must be positive")


def mapping_tensor(batch: StructuredBatch, device: torch.device) -> torch.Tensor:
    return torch.tensor(batch.mappings, dtype=torch.long, device=device)


def target_for_depth(batch: StructuredBatch, depth: int, device: torch.device) -> torch.Tensor:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    mapping = mapping_tensor(batch, device)
    state = batch.query_keys.to(device)
    rows = torch.arange(state.numel(), device=device)
    for _ in range(depth):
        state = mapping[rows, state]
    return state


def oracle_states_by_hop(
    batch: StructuredBatch, depth: int, device: torch.device
) -> list[torch.Tensor]:
    mapping = mapping_tensor(batch, device)
    state = batch.query_keys.to(device)
    rows = torch.arange(state.numel(), device=device)
    states = []
    for _ in range(depth):
        state = mapping[rows, state]
        states.append(state)
    return states


class RecurrentCompositionBinder(nn.Module):
    """Canonical identity codebook + differentiable recurrent binding."""

    def __init__(self, task: StructuredTaskConfig, cfg: CompositionConfig):
        super().__init__()
        cfg.validate(task)
        self.task = task
        self.cfg = cfg
        self.state = nn.Embedding(task.table_size, cfg.d_model)
        self.bind = AddressedPayloadBinding(cfg.d_model, cfg.d_model)
        self.payload_norm = RMSNorm(cfg.d_model)
        self.head = CosineTiedHead(task.table_size, cfg.d_model)
        self.head.weight = self.state.weight
        self._initialize_exact_geometry()

    @torch.no_grad()
    def _initialize_exact_geometry(self) -> None:
        self.state.weight.zero_()
        self.state.weight[:, : self.task.table_size].copy_(
            torch.eye(self.task.table_size, dtype=self.state.weight.dtype)
        )
        self.bind.q_proj.weight.copy_(torch.eye(self.cfg.d_model))
        self.bind.k_proj.weight.copy_(torch.eye(self.cfg.d_model))
        self.bind.logit_scale.fill_(math.log(self.cfg.initial_scale))
        self.state.weight.requires_grad_(False)
        self.bind.q_proj.weight.requires_grad_(False)
        self.bind.k_proj.weight.requires_grad_(False)
        self.payload_norm.weight.requires_grad_(False)
        # CosineTiedHead shares the frozen state Parameter.

    def encode_table(
        self, batch: StructuredBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        addresses = self.state(batch.row_sources)
        payloads = self.state(batch.row_values)
        query = self.state(batch.query_keys)
        return addresses, payloads, query

    def readout_logits(self, primary: torch.Tensor) -> torch.Tensor:
        return self.head(primary)

    def reason(
        self,
        batch: StructuredBatch,
        *,
        depth: int,
        mode: Mode,
        intervention: Intervention = "normal",
        return_history: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if mode not in ("recurrent", "static", "hard_recurrent"):
            raise ValueError(f"unknown mode: {mode}")
        if intervention not in ("normal", "corrupt_after_2"):
            raise ValueError(f"unknown intervention: {intervention}")

        addresses, payloads, query0 = self.encode_table(batch)
        query = query0
        primary: torch.Tensor | None = None
        history: list[dict] = []
        oracle = oracle_states_by_hop(batch, depth, query.device)
        batch_rows = torch.arange(query.size(0), device=query.device)

        for hop in range(depth):
            selected, attention = self.bind(
                query[:, None, :],
                addresses,
                payloads,
            )
            primary = self.payload_norm(selected[:, 0])

            current_source = batch.query_keys.to(query.device) if hop == 0 else oracle[hop - 1]
            expected = batch.row_sources.to(query.device).eq(current_source[:, None])
            if not torch.all(expected.sum(-1).eq(1)):
                raise AssertionError("every permutation table must contain each source exactly once")
            expected_index = expected.float().argmax(-1)
            correct_mass = attention[:, 0][batch_rows, expected_index]
            entropy = -(
                attention[:, 0].float().clamp_min(1e-30)
                * attention[:, 0].float().clamp_min(1e-30).log()
            ).sum(-1)
            oracle_code = self.state(oracle[hop])
            cosine = F.cosine_similarity(primary.float(), oracle_code.float(), dim=-1)

            if return_history:
                history.append(
                    {
                        "hop": hop + 1,
                        "primary": primary,
                        "query": query,
                        "attention": attention[:, 0],
                        "correct_address_hit": attention[:, 0].argmax(-1).eq(expected_index),
                        "correct_address_mass": correct_mass,
                        "address_entropy": entropy,
                        "oracle_state": oracle[hop],
                        "oracle_cosine": cosine,
                    }
                )

            if hop + 1 < depth:
                if mode == "static":
                    query = query0
                elif mode == "hard_recurrent":
                    next_id = self.readout_logits(primary).argmax(-1)
                    query = self.state(next_id)
                else:
                    query = primary

                # Causal intervention after exactly two completed hops.  Rolling
                # query states across examples keeps dimensions/distribution but
                # breaks example-specific trajectory identity.
                if intervention == "corrupt_after_2" and hop + 1 == 2:
                    query = query.roll(1, dims=0)

        if primary is None:
            raise AssertionError("positive depth must produce a primary state")
        return primary, {"history": history, "query0": query0}


def _aggregate_history(history: list[dict]) -> list[dict]:
    rows = []
    for item in history:
        rows.append(
            {
                "hop": int(item["hop"]),
                "address_hit_accuracy": float(item["correct_address_hit"].float().mean()),
                "correct_address_mass": float(item["correct_address_mass"].float().mean()),
                "address_entropy": float(item["address_entropy"].float().mean()),
                "oracle_cosine": float(item["oracle_cosine"].float().mean()),
                "minimum_oracle_cosine": float(item["oracle_cosine"].float().min()),
            }
        )
    return rows


@torch.inference_mode()
def evaluate_depth(
    model: RecurrentCompositionBinder,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: Literal["validation", "test"],
    depth: int,
    examples: int,
    batch_size: int,
    device: torch.device,
    mode: Mode,
    intervention: Intervention = "normal",
) -> dict:
    was_training = model.training
    model.eval()
    correct = 0
    nll = 0.0
    digest = hashlib.sha256()
    per_hop_sums: list[dict[str, float]] | None = None
    try:
        for start in range(0, examples, batch_size):
            count = min(batch_size, examples - start)
            batch = make_batch(
                task,
                data_seed=data_seed,
                split=split,
                start_index=start,
                batch_size=count,
            )
            _update_stream_digest(digest, batch)
            batch = batch.to(device)
            target = target_for_depth(batch, depth, device)
            primary, aux = model.reason(
                batch,
                depth=depth,
                mode=mode,
                intervention=intervention,
                return_history=True,
            )
            logits = model.readout_logits(primary)
            correct += int(logits.argmax(-1).eq(target).sum())
            nll += float(F.cross_entropy(logits.float(), target, reduction="sum"))
            aggregated = _aggregate_history(aux["history"])
            if per_hop_sums is None:
                per_hop_sums = [
                    {
                        "hop": row["hop"],
                        "address_hit_accuracy": 0.0,
                        "correct_address_mass": 0.0,
                        "address_entropy": 0.0,
                        "oracle_cosine": 0.0,
                        "minimum_oracle_cosine": 1.0,
                        "count": 0.0,
                    }
                    for row in aggregated
                ]
            for dst, src in zip(per_hop_sums, aggregated, strict=True):
                dst["address_hit_accuracy"] += src["address_hit_accuracy"] * count
                dst["correct_address_mass"] += src["correct_address_mass"] * count
                dst["address_entropy"] += src["address_entropy"] * count
                dst["oracle_cosine"] += src["oracle_cosine"] * count
                dst["minimum_oracle_cosine"] = min(
                    dst["minimum_oracle_cosine"], src["minimum_oracle_cosine"]
                )
                dst["count"] += count
    finally:
        model.train(was_training)

    hop_metrics = []
    for row in per_hop_sums or []:
        count = row.pop("count")
        hop_metrics.append(
            {
                **row,
                "address_hit_accuracy": row["address_hit_accuracy"] / count,
                "correct_address_mass": row["correct_address_mass"] / count,
                "address_entropy": row["address_entropy"] / count,
                "oracle_cosine": row["oracle_cosine"] / count,
            }
        )
    return {
        "split": split,
        "depth": depth,
        "mode": mode,
        "intervention": intervention,
        "correct": correct,
        "count": examples,
        "accuracy": correct / examples,
        "nll": nll / examples,
        "stream_sha256": digest.hexdigest(),
        "hops": hop_metrics,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def build_model(
    task: StructuredTaskConfig,
    cfg: CompositionConfig,
    *,
    device: torch.device,
) -> RecurrentCompositionBinder:
    return RecurrentCompositionBinder(task, cfg).to(device)


def validation_suite(
    model: RecurrentCompositionBinder,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    depths: tuple[int, ...],
    examples: int,
    batch_size: int,
    device: torch.device,
    split: Literal["validation", "test"],
) -> dict:
    modes = ("recurrent", "static", "hard_recurrent")
    evaluations = {
        mode: [
            evaluate_depth(
                model,
                task,
                data_seed=data_seed,
                split=split,
                depth=depth,
                examples=examples,
                batch_size=batch_size,
                device=device,
                mode=mode,  # type: ignore[arg-type]
            )
            for depth in depths
        ]
        for mode in modes
    }
    corrupt_depths = tuple(depth for depth in depths if depth >= 3)
    evaluations["recurrent_corrupt_after_2"] = [
        evaluate_depth(
            model,
            task,
            data_seed=data_seed,
            split=split,
            depth=depth,
            examples=examples,
            batch_size=batch_size,
            device=device,
            mode="recurrent",
            intervention="corrupt_after_2",
        )
        for depth in corrupt_depths
    ]
    return evaluations


def train(args: argparse.Namespace) -> dict:
    if args.evaluation_split == "test":
        raise ValueError("training mode is validation-only; use --evaluate-checkpoint for test")
    device = torch.device(args.device)
    task = StructuredTaskConfig(table_size=args.table_size)
    cfg = CompositionConfig(d_model=args.d_model, initial_scale=args.initial_scale)
    cfg.validate(task)
    protocol = verify_mapping_holdout(
        task,
        data_seed=args.data_seed,
        train_examples=max(args.steps * args.batch_size, 512),
        validation_examples=args.eval_examples,
        test_examples=args.eval_examples,
    )
    model = build_model(task, cfg, device=device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if [name for name, p in model.named_parameters() if p.requires_grad] != ["bind.logit_scale"]:
        raise AssertionError("EXP-001 minimal gate must train only bind.logit_scale")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    stream = hashlib.sha256()
    trace = []
    started = time.perf_counter()
    for step in range(args.steps):
        depth = 1 + (step % args.train_max_depth)
        batch = make_batch(
            task,
            data_seed=args.data_seed,
            split="train",
            start_index=step * args.batch_size,
            batch_size=args.batch_size,
        )
        _update_stream_digest(stream, batch)
        batch = batch.to(device)
        target = target_for_depth(batch, depth, device)
        optimizer.zero_grad(set_to_none=True)
        primary, _ = model.reason(batch, depth=depth, mode="recurrent")
        logits = model.readout_logits(primary)
        loss = F.cross_entropy(logits.float(), target)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            row = {
                "step": step + 1,
                "depth": depth,
                "loss": float(loss.detach()),
                "batch_accuracy": float(logits.argmax(-1).eq(target).float().mean()),
                "logit_scale": float(model.bind.logit_scale.detach().exp()),
            }
            trace.append(row)
            print(
                f"[EXP-001] step={step + 1:4d}/{args.steps} depth={depth} "
                f"loss={row['loss']:.5f} acc={100*row['batch_accuracy']:.1f}% "
                f"scale={row['logit_scale']:.3f}",
                flush=True,
            )

    depths = tuple(range(1, args.eval_max_depth + 1))
    evaluations = validation_suite(
        model,
        task,
        data_seed=args.data_seed,
        depths=depths,
        examples=args.eval_examples,
        batch_size=args.eval_batch_size,
        device=device,
        split="validation",
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "task_config": asdict(task),
        "composition_config": asdict(cfg),
        "protocol": protocol,
        "train_steps": args.steps,
        "train_max_depth": args.train_max_depth,
        "training_stream_sha256": stream.hexdigest(),
        "train_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "final_logit_scale": float(model.bind.logit_scale.detach().exp()),
        "trace": trace,
        "validation": evaluations,
        "test_split_touched": False,
    }
    checkpoint = args.output_dir / "exp_001.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "task_config": asdict(task),
            "composition_config": asdict(cfg),
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "metrics": metrics,
        },
        checkpoint,
    )
    _write_json(args.output_dir / "metrics.json", metrics)
    return metrics


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    if args.output is None:
        raise ValueError("--output is required for checkpoint evaluation")
    payload = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("checkpoint is not EXP-001")
    task = StructuredTaskConfig(**payload["task_config"])
    cfg = CompositionConfig(**payload["composition_config"])
    device = torch.device(args.device)
    model = build_model(task, cfg, device=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    depths = tuple(range(1, args.eval_max_depth + 1))
    result = {
        "experiment": EXPERIMENT_NAME,
        "mode": "checkpoint_only_evaluation",
        "checkpoint": str(args.evaluate_checkpoint),
        "split": args.evaluation_split,
        "depths": depths,
        "eval": validation_suite(
            model,
            task,
            data_seed=args.data_seed,
            depths=depths,
            examples=args.eval_examples,
            batch_size=args.eval_batch_size,
            device=device,
            split=args.evaluation_split,
        ),
    }
    _write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluate-checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/research/exp_001"))
    p.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    p.add_argument("--data-seed", type=int, default=20260814)
    p.add_argument("--table-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--initial-scale", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--train-max-depth", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=5e-2)
    p.add_argument("--eval-examples", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-max-depth", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    if args.evaluate_checkpoint is not None:
        result = evaluate_checkpoint(args)
        print(json.dumps(result, indent=2))
        return
    metrics = train(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "metrics.json"),
                "final_logit_scale": metrics["final_logit_scale"],
                "recurrent_accuracy_by_depth": [
                    row["accuracy"] for row in metrics["validation"]["recurrent"]
                ],
                "static_accuracy_by_depth": [
                    row["accuracy"] for row in metrics["validation"]["static"]
                ],
                "test_split_touched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
