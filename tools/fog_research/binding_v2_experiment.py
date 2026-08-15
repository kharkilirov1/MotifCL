#!/usr/bin/env python3
"""Matched operator-disjoint gate for binding-preserving FOG v2.

The task is inherited from ``matched_structured_lookup_experiment.py``.  This
script changes only the latent interface:

* the raw query code addresses shuffled rows through differentiable compare;
* the primary slot contains selected payload, never the query residual;
* that slot bypasses workspace mixing and memory compression;
* classification reads the primary latent directly, without answer-BOS.

No operator ID or target label enters the writer; labels supervise only the
final task loss.  Validation is the default.  Test evaluation is checkpoint
only and must be requested explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from fog_lmw.memory import PersistentLatentMemory
from fog_lmw.motifs import (
    AddressedPayloadBinding,
    ExpandCompressFFN,
    LowRankCompareSelectAggregate,
)
from matched_structured_lookup_experiment import (
    StructuredBatch,
    StructuredInputMixin,
    StructuredModelConfig,
    StructuredTaskConfig,
    _update_stream_digest,
    initialize_by_parameter_name,
    make_batch,
    target_deranged_indices,
    verify_mapping_holdout,
)


EXPERIMENT_NAME = "fog_binding_preserving_v2"
Intervention = Literal[
    "normal",
    "zero_primary",
    "target_deranged_primary",
    "query_deranged",
]


class BindingV2StructuredLookup(nn.Module, StructuredInputMixin):
    """Parallel latent workspace with a protected exact-binding carrier."""

    def __init__(self, task: StructuredTaskConfig, cfg: StructuredModelConfig):
        super().__init__()
        cfg.validate(task)
        self.task = task
        self.cfg = cfg
        self.key = nn.Embedding(task.table_size, cfg.d_model)
        self.value = nn.Embedding(task.table_size, cfg.d_model)
        self.row_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.query_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.slot_role = nn.Parameter(torch.empty(cfg.latent_slots, cfg.d_model))
        self.address_norm = RMSNorm(cfg.d_model)
        self.row_norm = RMSNorm(cfg.d_model)
        self.bind = AddressedPayloadBinding(cfg.d_model, cfg.compare_rank)
        self.payload_norm = RMSNorm(cfg.d_model)
        self.slot_mix = LowRankCompareSelectAggregate(cfg.d_model, cfg.compare_rank)
        self.workspace_ff = ExpandCompressFFN(cfg.d_model, cfg.planner_ff)
        self.memory = PersistentLatentMemory(
            cfg.d_model, cfg.compare_rank, cfg.memory_slots
        )
        self.readout_norm = RMSNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, task.table_size, bias=False)
        # Tied value coordinates make "the fact is present" an exact,
        # falsifiable statement instead of arbitrary downstream decodability.
        self.classifier.weight = self.value.weight

    def encode_rows_and_query(
        self, batch: StructuredBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        addresses = self.key(batch.row_sources) + self.row_type
        payloads = self.value(batch.row_values)
        # Raw query code is deliberately not contextualized.  It can address a
        # source, but cannot already contain its mapped destination.
        query = self.key(batch.query_keys) + self.query_type.squeeze(1)
        return addresses, payloads, query

    def reason(
        self,
        batch: StructuredBatch,
        *,
        query_override: torch.Tensor | None = None,
        return_history: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        addresses, payloads, raw_query = self.encode_rows_and_query(batch)
        query = raw_query if query_override is None else query_override
        memory: torch.Tensor | None = None
        primary: torch.Tensor | None = None
        history: list[dict] = []
        for step in range(self.cfg.reasoning_steps):
            address = self.address_norm(query)[:, None, :] + self.slot_role[None]
            selected, attention = self.bind(
                address,
                self.row_norm(addresses),
                payloads,
            )
            # Slot zero is payload-only; q never reaches its content residual.
            primary = self.payload_norm(selected[:, 0])
            if self.cfg.latent_slots == 1:
                latent = primary[:, None]
            else:
                auxiliary = address[:, 1:] + selected[:, 1:]
                mixed, slot_attention = self.slot_mix(auxiliary, auxiliary)
                auxiliary = self.workspace_ff(auxiliary + mixed)
                latent = torch.cat([primary[:, None], auxiliary], dim=1)
            memory, memory_stats = self.memory.forward_preserving(
                memory, latent, protected_slots=1
            )
            if return_history:
                history.append(
                    {
                        "step": step,
                        "primary": primary,
                        "latent": latent,
                        "memory": memory,
                        "binding_attention": attention,
                        "memory_stats": memory_stats,
                    }
                )
        if primary is None or memory is None:
            raise AssertionError("positive reasoning depth must create a primary slot")
        return memory, {
            "primary": primary,
            "query": query,
            "history": history,
        }

    def logits(
        self,
        batch: StructuredBatch,
        *,
        intervention: Intervention = "normal",
    ) -> torch.Tensor:
        query_override = None
        if intervention == "query_deranged":
            counterfactual_keys = (batch.query_keys + 1) % self.task.table_size
            query_override = (
                self.key(counterfactual_keys) + self.query_type.squeeze(1)
            )
        memory, aux = self.reason(batch, query_override=query_override)
        primary = aux["primary"]
        if intervention == "zero_primary":
            primary = torch.zeros_like(primary)
        elif intervention == "target_deranged_primary":
            donors = target_deranged_indices(batch.targets)
            primary = primary.index_select(0, donors)
        elif intervention not in ("normal", "query_deranged"):
            raise ValueError(f"unknown intervention: {intervention}")
        # Read the same protected carrier exposed at memory[:, 0].
        if intervention == "normal" and not torch.equal(memory[:, 0], aux["primary"]):
            raise AssertionError("protected primary slot was not preserved")
        return self.classifier(self.readout_norm(primary))


@torch.no_grad()
def build_model(
    task: StructuredTaskConfig,
    cfg: StructuredModelConfig,
    *,
    model_seed: int,
) -> BindingV2StructuredLookup:
    model = BindingV2StructuredLookup(task, cfg)
    initialize_by_parameter_name(model, seed=model_seed, std=cfg.initializer_range)
    if cfg.fixed_orthogonal_keys:
        model.key.weight.zero_()
        model.key.weight[:, : task.table_size].copy_(
            torch.eye(task.table_size, dtype=model.key.weight.dtype)
        )
        model.key.weight.requires_grad_(False)
        model.value.weight.zero_()
        value_start = task.table_size
        model.value.weight[:, value_start : value_start + task.table_size].copy_(
            torch.eye(task.table_size, dtype=model.value.weight.dtype)
        )
        model.value.weight.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate(
    model: BindingV2StructuredLookup,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: Literal["validation", "test"],
    examples: int,
    batch_size: int,
    device: torch.device,
    intervention: Intervention,
) -> dict:
    was_training = model.training
    model.eval()
    correct = 0
    nll = 0.0
    digest = hashlib.sha256()
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
            logits = model.logits(batch, intervention=intervention)
            correct += int(logits.argmax(dim=-1).eq(batch.targets).sum())
            nll += float(F.cross_entropy(logits.float(), batch.targets, reduction="sum"))
    finally:
        model.train(was_training)
    return {
        "split": split,
        "intervention": intervention,
        "correct": correct,
        "count": examples,
        "accuracy": correct / examples,
        "nll": nll / examples,
        "stream_sha256": digest.hexdigest(),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def train_one(
    task: StructuredTaskConfig,
    cfg: StructuredModelConfig,
    *,
    model_seed: int,
    data_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_examples: int,
    eval_batch_size: int,
    split: Literal["validation", "test"],
    output_dir: Path,
    device: torch.device,
    log_every: int,
) -> dict:
    model = build_model(task, cfg, model_seed=model_seed).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    stream = hashlib.sha256()
    trace = []
    started = time.perf_counter()
    for step in range(steps):
        batch = make_batch(
            task,
            data_seed=data_seed,
            split="train",
            start_index=step * batch_size,
            batch_size=batch_size,
        )
        _update_stream_digest(stream, batch)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(batch)
        loss = F.cross_entropy(logits.float(), batch.targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "batch_accuracy": float(
                    logits.detach().argmax(dim=-1).eq(batch.targets).float().mean()
                ),
                "grad_norm": float(grad_norm),
            }
            trace.append(row)
            print(
                f"[binding-v2 seed={model_seed}] {step + 1}/{steps} "
                f"loss={row['loss']:.4f} acc={100*row['batch_accuracy']:.1f}%",
                flush=True,
            )
    evaluations = {
        intervention: evaluate(
            model,
            task,
            data_seed=data_seed,
            split=split,
            examples=eval_examples,
            batch_size=eval_batch_size,
            device=device,
            intervention=intervention,
        )
        for intervention in (
            "normal",
            "zero_primary",
            "target_deranged_primary",
            "query_deranged",
        )
    }
    if len({row["stream_sha256"] for row in evaluations.values()}) != 1:
        raise AssertionError("interventions received different evaluation streams")
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_seconds": time.perf_counter() - started,
        "training_stream_sha256": stream.hexdigest(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trace": trace,
        "eval": evaluations,
    }
    seed_dir = output_dir / f"seed_{model_seed:04d}"
    checkpoint = seed_dir / "binding_v2.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "task_config": asdict(task),
            "model_config": asdict(cfg),
            "model_seed": model_seed,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "metrics": metrics,
        },
        checkpoint,
    )
    _write_json(seed_dir / "binding_v2.metrics.json", metrics)
    return metrics


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    payload = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("checkpoint is not a binding-v2 experiment")
    task = StructuredTaskConfig(**payload["task_config"])
    cfg = StructuredModelConfig(**payload["model_config"])
    model = build_model(task, cfg, model_seed=int(payload["model_seed"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    result = {
        "experiment": EXPERIMENT_NAME,
        "mode": "checkpoint_only_evaluation",
        "checkpoint": str(args.evaluate_checkpoint),
        "data_seed": args.data_seed,
        "split": args.evaluation_split,
        "eval": {
            intervention: evaluate(
                model,
                task,
                data_seed=args.data_seed,
                split=args.evaluation_split,
                examples=args.eval_examples,
                batch_size=args.eval_batch_size,
                device=device,
                intervention=intervention,
            )
            for intervention in (
                "normal",
                "zero_primary",
                "target_deranged_primary",
                "query_deranged",
            )
        },
    }
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluate-checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/binding_v2"))
    p.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-seed", type=int, default=20260812)
    p.add_argument("--table-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--latent-slots", type=int, default=4)
    p.add_argument("--reasoning-steps", type=int, default=2)
    p.add_argument("--compare-rank", type=int, default=16)
    p.add_argument("--planner-ff", type=int, default=128)
    p.add_argument("--memory-slots", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-examples", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--fixed-orthogonal-keys", action="store_true", default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    if args.evaluate_checkpoint is not None:
        if args.output is None:
            raise ValueError("--output is required for checkpoint evaluation")
        evaluate_checkpoint(args)
        return
    if args.evaluation_split == "test":
        raise ValueError("train mode is validation-only; use --evaluate-checkpoint for test")
    task = StructuredTaskConfig(table_size=args.table_size)
    cfg = StructuredModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=max(32, args.table_size + args.memory_slots + 4),
        latent_slots=args.latent_slots,
        reasoning_steps=args.reasoning_steps,
        compare_rank=args.compare_rank,
        planner_ff=args.planner_ff,
        memory_slots=args.memory_slots,
        n_reasoning_modes=1,
        dropout=0.0,
        fixed_orthogonal_keys=args.fixed_orthogonal_keys,
    )
    protocol = verify_mapping_holdout(
        task,
        data_seed=args.data_seed,
        train_examples=max(args.steps * args.batch_size, 512),
        validation_examples=args.eval_examples,
        test_examples=args.eval_examples,
    )
    results = [
        train_one(
            task,
            cfg,
            model_seed=seed,
            data_seed=args.data_seed,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_examples=args.eval_examples,
            eval_batch_size=args.eval_batch_size,
            split="validation",
            output_dir=args.output_dir,
            device=torch.device(args.device),
            log_every=args.log_every,
        )
        for seed in args.seeds
    ]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "task_config": asdict(task),
        "model_config": asdict(cfg),
        "protocol": protocol,
        "test_split_touched": False,
        "runs": results,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "output": str(args.output_dir / "summary.json"),
        "validation_accuracy": [row["eval"]["normal"]["accuracy"] for row in results],
    }, indent=2))


if __name__ == "__main__":
    main()
