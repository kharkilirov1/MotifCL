#!/usr/bin/env python3
"""Train the full 10M query-bound-v2 model on unseen token lookup tables.

Unlike the structured diagnostic, this gate goes through the production token
embedding, four-layer causal backbone, recurrent planner API, tied 8K LM head,
checkpoint loader, and direct-latent first-token path.  Table operators remain
mapping-disjoint.  The answer token is present in the prompt, but its position
changes with a shuffled row order, so the model must bind query key to value.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner
from fog_lmw.checkpoint import atomic_torch_save, sha256_file
from matched_structured_lookup_experiment import (
    StructuredTaskConfig,
    make_batch,
    target_deranged_indices,
    verify_mapping_holdout,
)


EXPERIMENT_NAME = "fog_10m_token_binding_v2"
ROW_ID = 30
VALUE_ID = 31
QUERY_ID = 32
KEY_BASE = 100
VALUE_BASE = 200
BOS_ID = 1


def token_batch(
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: str,
    start_index: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    structured = make_batch(
        task,
        data_seed=data_seed,
        split=split,  # type: ignore[arg-type]
        start_index=start_index,
        batch_size=batch_size,
    )
    rows = []
    for sources, values, query in zip(
        structured.row_sources.tolist(),
        structured.row_values.tolist(),
        structured.query_keys.tolist(),
        strict=True,
    ):
        prompt = []
        for source, value in zip(sources, values, strict=True):
            prompt.extend([ROW_ID, KEY_BASE + source, VALUE_ID, VALUE_BASE + value])
        prompt.extend([QUERY_ID, KEY_BASE + query])
        rows.append(prompt)
    prompt_ids = torch.tensor(rows, dtype=torch.long)
    targets = VALUE_BASE + structured.targets
    answer = torch.stack([torch.full_like(targets, BOS_ID), targets], dim=1)
    return prompt_ids, answer, structured.query_keys


def direct_logits(
    model: FOGLatentReasoner,
    prompt: torch.Tensor,
    *,
    reasoning_steps: int,
    intervention: str = "normal",
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    _, aux = model.reason(prompt, reasoning_steps=reasoning_steps)
    primary = aux["primary_latent"]
    if primary is None:
        raise AssertionError("v2 reasoner did not produce primary latent")
    if intervention == "zero":
        primary = torch.zeros_like(primary)
    elif intervention == "target_deranged":
        if targets is None:
            raise ValueError("targets required for derangement")
        primary = primary.index_select(0, target_deranged_indices(targets))
    elif intervention != "normal":
        raise ValueError(intervention)
    return model.direct_vocab_logits(primary)


@torch.inference_mode()
def evaluate(
    model: FOGLatentReasoner,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: str,
    examples: int,
    batch_size: int,
    reasoning_steps: int,
    device: torch.device,
    intervention: str,
) -> dict:
    model.eval()
    correct = 0
    nll = 0.0
    address_hits = 0
    address_mass = 0.0
    address_entropy = 0.0
    digest = hashlib.sha256()
    for start in range(0, examples, batch_size):
        count = min(batch_size, examples - start)
        prompt, answer, query_keys = token_batch(
            task,
            data_seed=data_seed,
            split=split,
            start_index=start,
            batch_size=count,
        )
        target = answer[:, 1]
        digest.update(prompt.numpy().tobytes())
        if intervention == "query_cyclic":
            prompt = prompt.clone()
            prompt[:, -1] = KEY_BASE + ((query_keys + 1) % task.table_size)
            mode = "normal"
        else:
            mode = intervention
        prompt = prompt.to(device)
        target = target.to(device)
        if intervention == "normal":
            _, aux = model.reason(
                prompt,
                reasoning_steps=reasoning_steps,
                return_diagnostics=True,
            )
            primary = aux["primary_latent"]
            logits = model.direct_vocab_logits(primary)
            if tuple(model.cfg.binding_offsets) == (2,):
                weights = aux["history"][-1]["planner"]["binding_attention"][:, 0]
                candidate_count = prompt.size(1) - 2
                weights = weights[:, :candidate_count]
                query_token = prompt[:, -1]
                expected = prompt[:, :-2].eq(query_token[:, None])
                if not torch.all(expected.sum(dim=-1).eq(1)):
                    raise AssertionError("exact binding gate expects one matching key")
                expected_index = expected.float().argmax(dim=-1)
                address_hits += int(weights.argmax(dim=-1).eq(expected_index).sum())
                address_mass += float(
                    weights[torch.arange(count, device=device), expected_index].sum()
                )
                address_entropy += float(
                    (-(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(-1)).sum()
                )
        else:
            logits = direct_logits(
                model,
                prompt,
                reasoning_steps=reasoning_steps,
                intervention=mode,
                targets=target,
            )
        correct += int(logits.argmax(-1).eq(target).sum())
        nll += float(F.cross_entropy(logits.float(), target, reduction="sum"))
    result = {
        "split": split,
        "intervention": intervention,
        "correct": correct,
        "count": examples,
        "accuracy": correct / examples,
        "nll": nll / examples,
        "stream_sha256": digest.hexdigest(),
    }
    if intervention == "normal" and tuple(model.cfg.binding_offsets) == (2,):
        result.update(
            {
                "address_hit_accuracy": address_hits / examples,
                "correct_address_mass": address_mass / examples,
                "address_entropy": address_entropy / examples,
            }
        )
    return result


@torch.inference_mode()
def oracle_vocab_copy(model: FOGLatentReasoner, *, batch_size: int = 256) -> dict:
    correct = 0
    minimum_margin = float("inf")
    for start in range(0, model.cfg.vocab_size, batch_size):
        token_ids = torch.arange(
            start,
            min(start + batch_size, model.cfg.vocab_size),
            device=model.token.weight.device,
        )
        primary = model.planner.bind_norm(model.token(token_ids))
        logits = model.direct_vocab_logits(primary)
        correct += int(logits.argmax(-1).eq(token_ids).sum())
        target_logits = logits[torch.arange(token_ids.numel(), device=logits.device), token_ids]
        masked = logits.clone()
        masked[torch.arange(token_ids.numel(), device=logits.device), token_ids] = -torch.inf
        minimum_margin = min(
            minimum_margin,
            float((target_logits - masked.max(-1).values).min()),
        )
    return {
        "correct": correct,
        "count": model.cfg.vocab_size,
        "accuracy": correct / model.cfg.vocab_size,
        "minimum_logit_margin": minimum_margin,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def train(args: argparse.Namespace) -> None:
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = FOGReasonerConfig(**source["model_config"])
    if config.architecture_version != "query_bound_v2":
        raise ValueError("init checkpoint must use query_bound_v2")
    config.reasoning_steps = args.reasoning_steps
    config.diversity_weight = 0.0
    model = FOGLatentReasoner(config)
    model.load_state_dict(source["model_state_dict"], strict=True)
    model.float().to(torch.device(args.device))
    if args.train_scope == "sharpness":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name == "planner.bind.logit_scale")
    elif args.train_scope == "binding":
        trainable_prefixes = (
            "planner.slot_role",
            "planner.binding_role",
            "planner.query_norm",
            "planner.context_norm",
            "planner.bind.",
            "planner.bind_norm",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise AssertionError("training scope selected no parameters")
    task = StructuredTaskConfig(table_size=args.table_size)
    protocol = verify_mapping_holdout(
        task,
        data_seed=args.data_seed,
        train_examples=args.steps * args.batch_size,
        validation_examples=args.eval_examples,
        test_examples=args.eval_examples,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    trace = []
    started = time.perf_counter()
    for step in range(args.steps):
        prompt, answer, _ = token_batch(
            task,
            data_seed=args.data_seed,
            split="train",
            start_index=step * args.batch_size,
            batch_size=args.batch_size,
        )
        prompt = prompt.to(args.device)
        answer = answer.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        target = answer[:, 1]
        logits = direct_logits(
            model, prompt, reasoning_steps=args.reasoning_steps
        )
        # This gate supervises only the exact first answer token.  Auxiliary
        # slot-diversity terms would otherwise update the address projections
        # without providing any evidence about query→payload binding.
        loss = F.cross_entropy(logits.float(), target)
        batch_accuracy = logits.argmax(-1).eq(target).float().mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "batch_accuracy": float(batch_accuracy.detach()),
                "grad_norm": float(grad_norm),
                "binding_score_scale": float(
                    model.planner.bind.logit_scale.detach().exp().clamp(max=200.0)
                ),
            }
            trace.append(row)
            print(
                f"[10m-token-v2] {step+1}/{args.steps} loss={row['loss']:.4f} "
                f"acc={100*row['batch_accuracy']:.1f}%",
                flush=True,
            )
    evaluations = {
        mode: evaluate(
            model,
            task,
            data_seed=args.data_seed,
            split="validation",
            examples=args.eval_examples,
            batch_size=args.eval_batch_size,
            reasoning_steps=args.reasoning_steps,
            device=torch.device(args.device),
            intervention=mode,
        )
        for mode in ("normal", "zero", "target_deranged", "query_cyclic")
    }
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "architecture_version": config.architecture_version,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "train_scope": args.train_scope,
        "init_checkpoint": str(args.init_checkpoint),
        "init_sha256": sha256_file(args.init_checkpoint),
        "data_seed": args.data_seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "reasoning_steps": args.reasoning_steps,
        "train_seconds": time.perf_counter() - started,
        "trace": trace,
        "eval": evaluations,
        "oracle_vocab_copy": oracle_vocab_copy(model),
        "protocol": protocol,
        "test_split_touched": False,
    }
    checkpoint = {
        "format_version": 2,
        "checkpoint_kind": "inference",
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "global_step": args.steps,
        "consumed_tokens": int(source.get("consumed_tokens", 0)),
        "tokenizer": source.get("tokenizer"),
        "metadata": {
            "stage": "binding-v2-token-lookup",
            "binding_examples": args.steps * args.batch_size,
            "metrics": metrics,
            "source_migration": source.get("metadata", {}),
        },
    }
    atomic_torch_save(checkpoint, args.output_checkpoint)
    _write_json(args.output_metrics, metrics)
    print(json.dumps({
        "checkpoint": str(args.output_checkpoint),
        "metrics": str(args.output_metrics),
        "validation": evaluations,
    }, indent=2))


def checkpoint_eval(args: argparse.Namespace) -> None:
    payload = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
    config = FOGReasonerConfig(**payload["model_config"])
    model = FOGLatentReasoner(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.float().to(torch.device(args.device))
    task = StructuredTaskConfig(table_size=args.table_size)
    result = {
        "experiment": EXPERIMENT_NAME,
        "split": args.evaluation_split,
        "checkpoint_sha256": sha256_file(args.evaluate_checkpoint),
        "eval": {
            mode: evaluate(
                model,
                task,
                data_seed=args.data_seed,
                split=args.evaluation_split,
                examples=args.eval_examples,
                batch_size=args.eval_batch_size,
                reasoning_steps=args.reasoning_steps,
                device=torch.device(args.device),
                intervention=mode,
            )
            for mode in ("normal", "zero", "target_deranged", "query_cyclic")
        },
    }
    _write_json(args.output_metrics, result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--evaluate-checkpoint", type=Path)
    p.add_argument("--output-checkpoint", type=Path, default=Path("checkpoints/fog_binding_v2_10m_token_lookup_fp32.pt"))
    p.add_argument("--output-metrics", type=Path, default=Path("artifacts/binding_v2_10m_token_lookup.json"))
    p.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    p.add_argument("--table-size", type=int, default=8)
    p.add_argument("--data-seed", type=int, default=20260813)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--reasoning-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--train-scope",
        choices=("sharpness", "binding", "all"),
        default="sharpness",
    )
    p.add_argument("--eval-examples", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    if args.evaluate_checkpoint:
        checkpoint_eval(args)
    else:
        if args.init_checkpoint is None:
            raise ValueError("--init-checkpoint is required for training")
        if args.evaluation_split == "test":
            raise ValueError("training is validation-only")
        train(args)


if __name__ == "__main__":
    main()
