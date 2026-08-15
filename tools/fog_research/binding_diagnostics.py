"""Binding diagnostics for the structured FOG lookup experiment.

This module deliberately does not modify :mod:`fog_lmw` or the matched gate.
It asks three narrower questions using validation-only, mapping-disjoint data:

1. Can a frozen linear or nonlinear probe recover the exact target from the
   latent proposals or persistent memory produced by a failed FOG checkpoint?
2. Does permutation-invariant row pooling destroy key/value binding while an
   explicit query-conditioned row selection preserves it?
3. Can the existing strict second-pass decoder learn to read a perfect oracle
   answer vector when the writer/planner/compressor are bypassed?

The first question separates "the fact is absent" from "the normal reader did
not use it".  The second is a constructive binding control.  The third tests
the reader interface independently of latent writing quality.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from matched_structured_lookup_experiment import (
    EXPERIMENT_NAME,
    FOGStructuredLookup,
    FOG_VARIANTS,
    StructuredBatch,
    StructuredModelConfig,
    StructuredTaskConfig,
    build_model,
    make_batch,
    target_deranged_indices,
)


BINDING_EXPERIMENT_NAME = "fog_structured_binding_diagnostics_v1"
ProbeKind = Literal["linear", "mlp"]


@dataclass(frozen=True)
class FrozenFeatureBank:
    """CPU feature tensors extracted from a frozen FOG writer."""

    features: dict[str, torch.Tensor]
    slots: dict[str, torch.Tensor]
    queries: torch.Tensor
    targets: torch.Tensor
    sample_indices: tuple[int, ...]
    split: str

    @property
    def count(self) -> int:
        return int(self.targets.numel())


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _state_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda row: row[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_digest(model: nn.Module) -> str:
    return _state_digest(model.state_dict().items())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def exact_query_row(
    rows: torch.Tensor,
    row_sources: torch.Tensor,
    query_keys: torch.Tensor,
) -> torch.Tensor:
    """Select the unique row whose source equals the query key.

    This is an oracle *address*, not an oracle answer: the returned continuous
    row must still be decoded into its value class by a probe.
    """

    if rows.ndim != 3 or row_sources.shape != rows.shape[:2]:
        raise ValueError("rows must be [batch, rows, d] with matching sources")
    if query_keys.shape != (rows.size(0),):
        raise ValueError("query_keys must have shape [batch]")
    matches = row_sources.eq(query_keys.unsqueeze(1))
    if not torch.all(matches.sum(dim=1).eq(1)):
        raise ValueError("every query must match exactly one source row")
    return rows[matches]


def dot_query_row(
    rows: torch.Tensor,
    query: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Content-address rows with a fixed dot-product query."""

    if rows.ndim != 3 or query.shape != (rows.size(0), rows.size(2)):
        raise ValueError("query must be [batch, d] and match rows")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = torch.einsum("bd,brd->br", query.float(), rows.float()) / temperature
    weights = scores.softmax(dim=-1).to(rows.dtype)
    return torch.einsum("br,brd->bd", weights, rows)


@torch.inference_mode()
def extract_frozen_features(
    model: FOGStructuredLookup,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: Literal["train", "validation"],
    examples: int,
    batch_size: int,
    device: torch.device,
) -> FrozenFeatureBank:
    """Capture proposals/memory from every reasoning step without gradients."""

    if examples <= 0 or batch_size <= 0:
        raise ValueError("examples and batch_size must be positive")
    before = _model_digest(model)
    was_training = model.training
    model.eval()
    feature_parts: dict[str, list[torch.Tensor]] = {}
    slot_parts: dict[str, list[torch.Tensor]] = {}
    query_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    sample_indices: list[int] = []
    try:
        for start in range(0, examples, batch_size):
            count = min(batch_size, examples - start)
            cpu_batch = make_batch(
                task,
                data_seed=data_seed,
                split=split,
                start_index=start,
                batch_size=count,
            )
            batch = cpu_batch.to(device)
            prompt = model.encode_prompt(batch)
            rows = prompt[:, : task.table_size]
            query = prompt[:, -1]

            raw_features = {
                # This representation is invariant to the mapping permutation:
                # every key and every value occurs exactly once.
                "raw_row_mean_plus_query": torch.cat(
                    [rows.mean(dim=1), query], dim=-1
                ),
                "raw_exact_query_row": exact_query_row(
                    rows, batch.row_sources, batch.query_keys
                ),
                "raw_dot_query_row": dot_query_row(rows, query),
            }
            for name, tensor in raw_features.items():
                feature_parts.setdefault(name, []).append(tensor.cpu())

            memory: torch.Tensor | None = None
            for step in range(model.cfg.reasoning_steps):
                hidden, context_mask = model._backbone_with_memory(prompt, memory)
                step_number = step + 1
                # The original query position is the last prompt position.  On
                # recurrent passes ``hidden[:, -1]`` instead refers to the last
                # appended memory slot, so retain both localization points.
                feature_parts.setdefault(
                    f"context_query_step_{step_number}", []
                ).append(hidden[:, task.prompt_length - 1].cpu())
                feature_parts.setdefault(
                    f"context_last_step_{step_number}", []
                ).append(hidden[:, -1].cpu())
                proposal, _ = model.planner(hidden, context_mask=context_mask)
                memory, _ = model.memory(memory, proposal)
                for stem, tensor in (
                    (f"proposal_step_{step_number}", proposal),
                    (f"memory_step_{step_number}", memory),
                ):
                    slot_parts.setdefault(stem, []).append(tensor.cpu())
                    feature_parts.setdefault(f"{stem}_mean", []).append(
                        tensor.mean(dim=1).cpu()
                    )
                    feature_parts.setdefault(f"{stem}_flat", []).append(
                        tensor.flatten(start_dim=1).cpu()
                    )

            query_parts.append(query.cpu())
            target_parts.append(batch.targets.cpu())
            sample_indices.extend(cpu_batch.sample_indices)
    finally:
        model.train(was_training)
    after = _model_digest(model)
    if before != after:
        raise AssertionError("frozen feature extraction mutated the writer")
    return FrozenFeatureBank(
        features={name: torch.cat(parts) for name, parts in feature_parts.items()},
        slots={name: torch.cat(parts) for name, parts in slot_parts.items()},
        queries=torch.cat(query_parts),
        targets=torch.cat(target_parts),
        sample_indices=tuple(sample_indices),
        split=split,
    )


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryConditionedSlotProbe(nn.Module):
    """Learned address over frozen latent slots, followed by a small reader."""

    def __init__(self, d_model: int, rank: int, hidden_dim: int, classes: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.slot_norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, rank, bias=False)
        self.key_proj = nn.Linear(d_model, rank, bias=False)
        self.reader = MLPProbe(2 * d_model, hidden_dim, classes)
        self.rank = rank

    def forward(
        self, slots: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(self.query_norm(query))
        k = self.key_proj(self.slot_norm(slots))
        weights = torch.einsum("bd,bsd->bs", q, k).div(math.sqrt(self.rank))
        weights = weights.softmax(dim=-1)
        selected = torch.einsum("bs,bsd->bd", weights, slots)
        logits = self.reader(torch.cat([selected, query], dim=-1))
        return logits, weights


@torch.inference_mode()
def _evaluate_feature_probe(
    probe: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    probe.eval()
    correct = 0
    nll = 0.0
    predictions: list[torch.Tensor] = []
    observed_targets: list[torch.Tensor] = []
    for start in range(0, targets.numel(), batch_size):
        x = features[start : start + batch_size].to(device)
        y = targets[start : start + batch_size].to(device)
        logits = probe(x)
        predicted = logits.argmax(dim=-1)
        correct += int(predicted.eq(y).sum())
        nll += float(F.cross_entropy(logits.float(), y, reduction="sum"))
        predictions.append(predicted.cpu())
        observed_targets.append(y.cpu())
    return _classification_report(
        torch.cat(predictions),
        torch.cat(observed_targets),
        nll=nll,
    )


def _classification_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    nll: float,
) -> dict:
    if predictions.shape != targets.shape or targets.ndim != 1:
        raise ValueError("predictions and targets must be matching 1D tensors")
    classes = int(targets.max()) + 1
    class_rows = []
    recalls = []
    for label in range(classes):
        mask = targets.eq(label)
        count = int(mask.sum())
        class_correct = int(predictions[mask].eq(label).sum())
        recall = class_correct / count if count else float("nan")
        class_rows.append(
            {"label": label, "count": count, "correct": class_correct, "recall": recall}
        )
        if count:
            recalls.append(recall)
    correct = int(predictions.eq(targets).sum())
    return {
        "accuracy": correct / targets.numel(),
        "correct": correct,
        "count": int(targets.numel()),
        "nll": nll / targets.numel(),
        "macro_accuracy": sum(recalls) / len(recalls),
        "per_class": class_rows,
    }


def train_feature_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    kind: ProbeKind,
    classes: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    seed: int,
    device: torch.device,
) -> dict:
    if kind == "linear":
        probe: nn.Module = nn.Linear(train_features.size(1), classes)
    elif kind == "mlp":
        probe = MLPProbe(train_features.size(1), hidden_dim, classes)
    else:
        raise ValueError(f"unknown probe kind: {kind}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    probe.to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    probe.train()
    last_loss = float("nan")
    for _ in range(steps):
        indices = torch.randint(
            train_targets.numel(),
            (min(batch_size, train_targets.numel()),),
            generator=generator,
        )
        x = train_features.index_select(0, indices).to(device)
        y = train_targets.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return {
        "kind": kind,
        "input_dim": train_features.size(1),
        "parameters": sum(p.numel() for p in probe.parameters()),
        "steps": steps,
        "last_minibatch_loss": last_loss,
        "train": _evaluate_feature_probe(
            probe, train_features, train_targets, device=device
        ),
        "validation": _evaluate_feature_probe(
            probe, validation_features, validation_targets, device=device
        ),
    }


@torch.inference_mode()
def _evaluate_slot_probe(
    probe: QueryConditionedSlotProbe,
    slots: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    probe.eval()
    correct = 0
    nll = 0.0
    entropy = 0.0
    predictions: list[torch.Tensor] = []
    observed_targets: list[torch.Tensor] = []
    for start in range(0, targets.numel(), batch_size):
        z = slots[start : start + batch_size].to(device)
        q = queries[start : start + batch_size].to(device)
        y = targets[start : start + batch_size].to(device)
        logits, weights = probe(z, q)
        predicted = logits.argmax(dim=-1)
        correct += int(predicted.eq(y).sum())
        nll += float(F.cross_entropy(logits.float(), y, reduction="sum"))
        entropy += float(
            (-(weights.float() * weights.float().clamp_min(1e-9).log()).sum(-1)).sum()
        )
        predictions.append(predicted.cpu())
        observed_targets.append(y.cpu())
    report = _classification_report(
        torch.cat(predictions),
        torch.cat(observed_targets),
        nll=nll,
    )
    report["mean_attention_entropy"] = entropy / targets.numel()
    return report


def train_slot_probe(
    train_slots: torch.Tensor,
    train_queries: torch.Tensor,
    train_targets: torch.Tensor,
    validation_slots: torch.Tensor,
    validation_queries: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    classes: int,
    rank: int,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    probe = QueryConditionedSlotProbe(
        train_slots.size(-1), rank, hidden_dim, classes
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    last_loss = float("nan")
    for _ in range(steps):
        indices = torch.randint(
            train_targets.numel(),
            (min(batch_size, train_targets.numel()),),
            generator=generator,
        )
        z = train_slots.index_select(0, indices).to(device)
        q = train_queries.index_select(0, indices).to(device)
        y = train_targets.index_select(0, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = probe(z, q)
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return {
        "kind": "learned_query_conditioned_slot_attention_mlp",
        "slots": train_slots.size(1),
        "d_model": train_slots.size(2),
        "parameters": sum(p.numel() for p in probe.parameters()),
        "steps": steps,
        "last_minibatch_loss": last_loss,
        "train": _evaluate_slot_probe(
            probe,
            train_slots,
            train_queries,
            train_targets,
            device=device,
        ),
        "validation": _evaluate_slot_probe(
            probe,
            validation_slots,
            validation_queries,
            validation_targets,
            device=device,
        ),
    }


def oracle_answer_memory(
    targets: torch.Tensor,
    *,
    table_size: int,
    d_model: int,
    memory_slots: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a fixed, exact one-hot answer code repeated across memory slots."""

    if d_model < table_size or memory_slots <= 0:
        raise ValueError("oracle code requires d_model >= table_size and slots > 0")
    codebook = torch.zeros(table_size, d_model, dtype=dtype, device=device)
    codebook[:, :table_size] = torch.eye(
        table_size, dtype=dtype, device=device
    ) * math.sqrt(d_model)
    return codebook.index_select(0, targets).unsqueeze(1).expand(-1, memory_slots, -1)


def _reader_trainable(name: str) -> bool:
    return name.startswith(("backbone.", "classifier.", "answer_bos", "neutral"))


@torch.inference_mode()
def evaluate_oracle_reader(
    model: FOGStructuredLookup,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    examples: int,
    batch_size: int,
    device: torch.device,
    intervention: Literal["normal", "zero", "target_deranged_shuffle"],
) -> dict:
    model.eval()
    correct = 0
    nll = 0.0
    for start in range(0, examples, batch_size):
        count = min(batch_size, examples - start)
        batch = make_batch(
            task,
            data_seed=data_seed,
            split="validation",
            start_index=start,
            batch_size=count,
        ).to(device)
        prompt = model.encode_prompt(batch)
        memory = oracle_answer_memory(
            batch.targets,
            table_size=task.table_size,
            d_model=model.cfg.d_model,
            memory_slots=model.cfg.memory_slots,
            dtype=prompt.dtype,
            device=device,
        )
        if intervention == "zero":
            memory = torch.zeros_like(memory)
        elif intervention == "target_deranged_shuffle":
            donors = target_deranged_indices(batch.targets)
            memory = memory.index_select(0, donors)
        elif intervention != "normal":
            raise ValueError(f"unknown intervention: {intervention}")
        logits = model.decode_embeds("fog_strict", prompt, memory)
        correct += int(logits.argmax(dim=-1).eq(batch.targets).sum())
        nll += float(F.cross_entropy(logits.float(), batch.targets, reduction="sum"))
    return {
        "intervention": intervention,
        "accuracy": correct / examples,
        "correct": correct,
        "count": examples,
        "nll": nll / examples,
    }


def train_oracle_reader(
    source_model: FOGStructuredLookup,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    validation_examples: int,
    validation_batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Train only the second-pass reader on perfect answer-coded memory."""

    before_training = {
        intervention: evaluate_oracle_reader(
            source_model,
            task,
            data_seed=data_seed,
            examples=validation_examples,
            batch_size=validation_batch_size,
            device=device,
            intervention=intervention,
        )
        for intervention in ("normal", "zero", "target_deranged_shuffle")
    }
    # Work on an independent clone so probe extraction and its source checkpoint
    # remain immutable.  Building before loading also preserves strict config.
    clone = build_model(
        "fog_strict", task, source_model.cfg, model_seed=seed
    )
    if not isinstance(clone, FOGStructuredLookup):
        raise AssertionError("fog_strict must build FOGStructuredLookup")
    clone.load_state_dict(source_model.state_dict(), strict=True)
    clone.to(device)
    frozen_before: dict[str, torch.Tensor] = {}
    trainable_names: list[str] = []
    for name, parameter in clone.named_parameters():
        parameter.requires_grad_(_reader_trainable(name))
        if parameter.requires_grad:
            trainable_names.append(name)
        else:
            frozen_before[name] = parameter.detach().cpu().clone()
    trainable = [p for p in clone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    train_correct = 0
    train_count = 0
    last_loss = float("nan")
    started = time.perf_counter()
    for step in range(steps):
        batch = make_batch(
            task,
            data_seed=data_seed,
            split="train",
            start_index=step * batch_size,
            batch_size=batch_size,
        ).to(device)
        # Strict decoding ignores prompt contents, but encode it to exercise the
        # exact production method and sequence geometry.
        with torch.no_grad():
            prompt = clone.encode_prompt(batch)
            memory = oracle_answer_memory(
                batch.targets,
                table_size=task.table_size,
                d_model=clone.cfg.d_model,
                memory_slots=clone.cfg.memory_slots,
                dtype=prompt.dtype,
                device=device,
            )
        clone.train()
        optimizer.zero_grad(set_to_none=True)
        logits = clone.decode_embeds("fog_strict", prompt, memory)
        loss = F.cross_entropy(logits.float(), batch.targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        train_correct += int(logits.detach().argmax(dim=-1).eq(batch.targets).sum())
        train_count += batch_size
        last_loss = float(loss.detach())

    for name, parameter in clone.named_parameters():
        if name in frozen_before and not torch.equal(
            parameter.detach().cpu(), frozen_before[name]
        ):
            raise AssertionError(f"oracle reader changed frozen writer tensor {name}")
    evaluations = {
        intervention: evaluate_oracle_reader(
            clone,
            task,
            data_seed=data_seed,
            examples=validation_examples,
            batch_size=validation_batch_size,
            device=device,
            intervention=intervention,
        )
        for intervention in ("normal", "zero", "target_deranged_shuffle")
    }
    return {
        "oracle_code": "sqrt(d_model) * one_hot(target), repeated over memory slots",
        "writer_path_bypassed": True,
        "before_reader_adaptation": before_training,
        "planner_compressor_and_input_embeddings_verified_unchanged": True,
        "shared_backbone_is_adapted_as_reader": True,
        "trainable_scope": "backbone + classifier + answer_bos + neutral",
        "trainable_tensor_names": trainable_names,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "steps": steps,
        "batch_size": batch_size,
        "last_minibatch_loss": last_loss,
        "train_online_accuracy": train_correct / train_count,
        "seconds": time.perf_counter() - started,
        "validation": evaluations,
    }


def load_fog_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[FOGStructuredLookup, StructuredTaskConfig, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(f"not a {EXPERIMENT_NAME} checkpoint")
    variant = payload.get("variant")
    if variant not in FOG_VARIANTS:
        raise ValueError("binding diagnostics require a FOG checkpoint")
    task = StructuredTaskConfig(**payload["task_config"])
    cfg = StructuredModelConfig(**payload["model_config"])
    model_seed = int(payload.get("metrics", {}).get("model_seed", 0))
    model = build_model(variant, task, cfg, model_seed=model_seed)
    if not isinstance(model, FOGStructuredLookup):
        raise AssertionError("FOG checkpoint did not build FOGStructuredLookup")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, task, {
        "path": str(checkpoint_path),
        "sha256": _file_sha256(checkpoint_path),
        "variant": variant,
        "model_seed": model_seed,
        "training_stream_sha256": payload.get("metrics", {}).get(
            "training_stream_sha256"
        ),
        "model_config": asdict(cfg),
        "task_config": asdict(task),
    }


def _feature_invariance_report(bank: FrozenFeatureBank) -> dict:
    row_mean = bank.features["raw_row_mean_plus_query"][:, : bank.queries.size(1)]
    # Mapping/order should cancel analytically; float summation order may leave
    # sub-ULP noise, hence report rather than demand bit identity.
    deviations = row_mean - row_mean[0]
    return {
        "raw_row_mean_max_abs_deviation": float(deviations.abs().max()),
        "explanation": (
            "sum_i(key_i + value_perm(i)) is constant across permutations; "
            "only the separately concatenated query varies"
        ),
    }


def _label_distribution(targets: torch.Tensor, classes: int) -> dict:
    counts = torch.bincount(targets, minlength=classes)
    maximum = int(counts.max())
    return {
        "counts": [int(value) for value in counts],
        "count": int(targets.numel()),
        "empirical_majority_accuracy": maximum / targets.numel(),
        "uniform_chance_accuracy": 1.0 / classes,
    }


def run_checkpoint_diagnostics(
    checkpoint_path: Path,
    *,
    data_seed: int,
    train_examples: int,
    validation_examples: int,
    extraction_batch_size: int,
    probe_steps: int,
    probe_batch_size: int,
    probe_learning_rate: float,
    probe_hidden_dim: int,
    oracle_steps: int,
    oracle_learning_rate: float,
    oracle_batch_size: int,
    device: torch.device,
) -> dict:
    model, task, checkpoint = load_fog_checkpoint(checkpoint_path, device)
    started = time.perf_counter()
    train_bank = extract_frozen_features(
        model,
        task,
        data_seed=data_seed,
        split="train",
        examples=train_examples,
        batch_size=extraction_batch_size,
        device=device,
    )
    validation_bank = extract_frozen_features(
        model,
        task,
        data_seed=data_seed,
        split="validation",
        examples=validation_examples,
        batch_size=extraction_batch_size,
        device=device,
    )
    if set(train_bank.features) != set(validation_bank.features):
        raise AssertionError("train/validation feature sets differ")

    # Fixed compact matrix: raw binding controls, localization points around
    # each writer step, and both pooled/flattened final latent states.
    last_step = model.cfg.reasoning_steps
    selected_features = [
        "raw_row_mean_plus_query",
        "raw_exact_query_row",
        "raw_dot_query_row",
    ]
    for step in range(1, last_step + 1):
        selected_features.extend(
            [
                f"context_query_step_{step}",
                f"context_last_step_{step}",
                f"proposal_step_{step}_flat",
                f"memory_step_{step}_flat",
            ]
        )
    selected_features.extend(
        [
            f"proposal_step_{last_step}_mean",
            f"memory_step_{last_step}_mean",
        ]
    )
    probes: dict[str, dict] = {}
    for feature_name in selected_features:
        for kind in ("linear", "mlp"):
            seed = _stable_seed(
                BINDING_EXPERIMENT_NAME,
                checkpoint["sha256"],
                feature_name,
                kind,
            ) % (2**31 - 1)
            probes[f"{feature_name}.{kind}"] = train_feature_probe(
                train_bank.features[feature_name],
                train_bank.targets,
                validation_bank.features[feature_name],
                validation_bank.targets,
                kind=kind,
                classes=task.table_size,
                steps=probe_steps,
                batch_size=probe_batch_size,
                learning_rate=probe_learning_rate,
                hidden_dim=probe_hidden_dim,
                seed=seed,
                device=device,
            )

    slot_name = f"memory_step_{last_step}"
    attention_seed = _stable_seed(
        BINDING_EXPERIMENT_NAME,
        checkpoint["sha256"],
        slot_name,
        "query_attention",
    ) % (2**31 - 1)
    probes[f"{slot_name}.query_conditioned_attention"] = train_slot_probe(
        train_bank.slots[slot_name],
        train_bank.queries,
        train_bank.targets,
        validation_bank.slots[slot_name],
        validation_bank.queries,
        validation_bank.targets,
        classes=task.table_size,
        rank=min(32, model.cfg.d_model),
        hidden_dim=probe_hidden_dim,
        steps=probe_steps,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        seed=attention_seed,
        device=device,
    )

    oracle_seed = _stable_seed(
        BINDING_EXPERIMENT_NAME, checkpoint["sha256"], "oracle_reader"
    ) % (2**31 - 1)
    oracle = train_oracle_reader(
        model,
        task,
        data_seed=data_seed,
        steps=oracle_steps,
        batch_size=oracle_batch_size,
        learning_rate=oracle_learning_rate,
        validation_examples=validation_examples,
        validation_batch_size=extraction_batch_size,
        seed=oracle_seed,
        device=device,
    )
    return {
        "checkpoint": checkpoint,
        "data": {
            "data_seed": data_seed,
            "train_split": "train",
            "validation_split": "validation",
            "test_split_touched": False,
            "train_examples": train_examples,
            "validation_examples": validation_examples,
            "mapping_partition": "inherited blake2b(mapping) % 10 holdout",
            "train_labels": _label_distribution(
                train_bank.targets, task.table_size
            ),
            "validation_labels": _label_distribution(
                validation_bank.targets, task.table_size
            ),
        },
        "chance_accuracy": task.chance_accuracy,
        "feature_invariance": _feature_invariance_report(train_bank),
        "probes": probes,
        "oracle_reader": oracle,
        "seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("binding_diagnostics_results.json")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=1729)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--extraction-batch-size", type=int, default=128)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    parser.add_argument("--oracle-steps", type=int, default=300)
    parser.add_argument("--oracle-batch-size", type=int, default=64)
    parser.add_argument("--oracle-learning-rate", type=float, default=1e-2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    random.seed(0)
    device = torch.device(args.device)
    runs = []
    for checkpoint in args.checkpoint:
        print(f"[binding] checkpoint={checkpoint}", flush=True)
        run = run_checkpoint_diagnostics(
            checkpoint,
            data_seed=args.data_seed,
            train_examples=args.train_examples,
            validation_examples=args.validation_examples,
            extraction_batch_size=args.extraction_batch_size,
            probe_steps=args.probe_steps,
            probe_batch_size=args.probe_batch_size,
            probe_learning_rate=args.probe_learning_rate,
            probe_hidden_dim=args.probe_hidden_dim,
            oracle_steps=args.oracle_steps,
            oracle_learning_rate=args.oracle_learning_rate,
            oracle_batch_size=args.oracle_batch_size,
            device=device,
        )
        runs.append(run)
        final_step = run["checkpoint"]["model_config"]["reasoning_steps"]
        memory_probe = run["probes"][f"memory_step_{final_step}_flat.mlp"]
        print(
            "[binding] "
            f"variant={run['checkpoint']['variant']} "
            f"memory_mlp={memory_probe['validation']['accuracy']:.4f} "
            f"oracle={run['oracle_reader']['validation']['normal']['accuracy']:.4f}",
            flush=True,
        )
    result = {
        "experiment": BINDING_EXPERIMENT_NAME,
        "mode": "validation_only_diagnostic",
        "test_split_touched": False,
        "runs": runs,
    }
    _write_json(args.output, result)
    print(f"[binding] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
