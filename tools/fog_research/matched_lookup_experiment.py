"""Matched permutation-lookup experiment for the FOG latent interface.

The task is deliberately algorithmic and distributionally stationary.  Every
example contains a freshly sampled permutation of one shared state-token set,
followed by a query state.  No fixed association can be memorized across
examples.  Table rows are serialized explicitly as

    <row> <key_i> <maps_to> <value_j> <row_end>

and the target is the mapped state token selected by the query.  A legacy
separate-key/value-token calibration remains available as an explicit ablation.

Three conditions share exactly the same lexical embedding, decoder backbone,
LM head, training batches, and name-derived initial weights:

* ``direct``: an ordinary causal decoder sees the complete serialized prompt;
* ``fog_full``: FOG reasons over the prompt and the final decoder sees both the
  complete prompt and latent memory;
* ``fog_strict``: FOG reasons over the prompt but the final decoder sees only a
  constant task token plus latent memory.

FOG evaluation additionally zeros or shuffles memory across examples.  Train,
validation, and test are deterministic and operator-disjoint: a hash of the
mapping permutation alone assigns the complete operator to an 80/10/10 split
before row order or query are sampled.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Iterable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner
from fog_lmw.checkpoint import atomic_torch_save
from fog_lmw.core import TinyDecoderBackbone


Variant = Literal["direct", "fog_full", "fog_strict"]
Split = Literal["train", "validation", "test"]
Intervention = Literal["normal", "zero", "shuffle"]

PAD = 0
ANSWER_BOS = 1
TASK = 2
ROW = 3
MAPS_TO = 4
ROW_END = 5
TABLE_END = 6
QUERY = 7
KEY_BASE = 8

VARIANTS: tuple[Variant, ...] = ("direct", "fog_full", "fog_strict")
FOG_VARIANTS: tuple[Variant, ...] = ("fog_full", "fog_strict")
EXPERIMENT_NAME = "matched_permutation_lookup_v2"


@dataclass(frozen=True)
class LookupTaskConfig:
    table_size: int = 8
    separate_key_value_tokens: bool = False

    def validate(self) -> None:
        # For n<3 one of the hash-partitioned splits can have no support.
        if self.table_size < 3:
            raise ValueError("table_size must be >= 3")

    @property
    def value_base(self) -> int:
        return KEY_BASE + self.table_size if self.separate_key_value_tokens else KEY_BASE

    @property
    def key_end(self) -> int:
        return KEY_BASE + self.table_size

    @property
    def vocab_size(self) -> int:
        return self.value_base + self.table_size

    @property
    def prompt_length(self) -> int:
        # TASK + n * (ROW key MAPS_TO value ROW_END) + TABLE_END QUERY key
        return 1 + 5 * self.table_size + 3

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.table_size

    def key_token(self, key: int) -> int:
        if not 0 <= key < self.table_size:
            raise ValueError("key is outside the table")
        return KEY_BASE + key

    def value_token(self, value: int) -> int:
        if not 0 <= value < self.table_size:
            raise ValueError("value is outside the table")
        return self.value_base + value


@dataclass(frozen=True)
class LookupExample:
    prompt_ids: tuple[int, ...]
    target_id: int
    mapping: tuple[int, ...]
    row_order: tuple[int, ...]
    query_key: int
    sample_index: int
    split: Split

    @property
    def signature(self) -> tuple[int, ...]:
        return (*self.prompt_ids, self.target_id)


@dataclass(frozen=True)
class LookupBatch:
    prompt_ids: torch.Tensor
    target_ids: torch.Tensor
    sample_indices: tuple[int, ...]
    signatures: tuple[tuple[int, ...], ...]
    mappings: tuple[tuple[int, ...], ...]

    def to(self, device: torch.device) -> "LookupBatch":
        return LookupBatch(
            prompt_ids=self.prompt_ids.to(device),
            target_ids=self.target_ids.to(device),
            sample_indices=self.sample_indices,
            signatures=self.signatures,
            mappings=self.mappings,
        )


class DirectLookupTransformer(nn.Module):
    """Ordinary decoder with FOG-compatible shared module names and geometry."""

    def __init__(self, cfg: FOGReasonerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = TinyDecoderBackbone(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

    def logits(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        bos = torch.full(
            (prompt_ids.size(0), 1),
            ANSWER_BOS,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        ids = torch.cat([prompt_ids, bos], dim=1)
        embeddings = self.token(ids)
        kinds = torch.zeros_like(ids)
        hidden = self.backbone.forward_embeds(embeddings, kinds)
        return self.lm_head(hidden[:, -1])


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _signature_bytes(signature: Iterable[int]) -> bytes:
    # JSON avoids assumptions about token width and is stable across platforms.
    return json.dumps(list(signature), separators=(",", ":")).encode("ascii")


def _mapping_bucket(mapping: Iterable[int]) -> int:
    digest = hashlib.blake2b(_signature_bytes(mapping), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 10


def _mapping_in_split(mapping: Iterable[int], split: Split) -> bool:
    bucket = _mapping_bucket(mapping)
    return (
        (split == "train" and bucket < 8)
        or (split == "validation" and bucket == 8)
        or (split == "test" and bucket == 9)
    )


def _candidate_example(
    task: LookupTaskConfig,
    *,
    data_seed: int,
    split: Split,
    sample_index: int,
    attempt: int,
) -> LookupExample:
    rng = random.Random(
        _stable_int("matched-permutation-lookup-v1", data_seed, split, sample_index, attempt)
    )
    mapping = list(range(task.table_size))
    row_order = list(range(task.table_size))
    rng.shuffle(mapping)
    rng.shuffle(row_order)
    query_key = rng.randrange(task.table_size)

    prompt = [TASK]
    for key in row_order:
        prompt.extend(
            [
                ROW,
                task.key_token(key),
                MAPS_TO,
                task.value_token(mapping[key]),
                ROW_END,
            ]
        )
    prompt.extend([TABLE_END, QUERY, task.key_token(query_key)])
    target = task.value_token(mapping[query_key])
    return LookupExample(
        prompt_ids=tuple(prompt),
        target_id=target,
        mapping=tuple(mapping),
        row_order=tuple(row_order),
        query_key=query_key,
        sample_index=sample_index,
        split=split,
    )


def make_example(
    task: LookupTaskConfig,
    *,
    data_seed: int,
    split: Split,
    sample_index: int,
) -> LookupExample:
    """Return deterministic sample ``sample_index`` from a disjoint split.

    Split assignment depends only on the mapping permutation.  Consequently an
    operator cannot cross splits under a different query or table row order.
    """

    task.validate()
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")
    for attempt in range(10_000):
        example = _candidate_example(
            task,
            data_seed=data_seed,
            split=split,
            sample_index=sample_index,
            attempt=attempt,
        )
        if _mapping_in_split(example.mapping, split):
            return example
    raise RuntimeError("could not draw an example in the requested hash split")


def make_batch(
    task: LookupTaskConfig,
    *,
    data_seed: int,
    split: Split,
    start_index: int,
    batch_size: int,
) -> LookupBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    examples = [
        make_example(
            task,
            data_seed=data_seed,
            split=split,
            sample_index=start_index + offset,
        )
        for offset in range(batch_size)
    ]
    return LookupBatch(
        prompt_ids=torch.tensor([row.prompt_ids for row in examples], dtype=torch.long),
        target_ids=torch.tensor([row.target_id for row in examples], dtype=torch.long),
        sample_indices=tuple(row.sample_index for row in examples),
        signatures=tuple(row.signature for row in examples),
        mappings=tuple(row.mapping for row in examples),
    )


def model_config(
    task: LookupTaskConfig,
    *,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    latent_slots: int = 4,
    reasoning_steps: int = 2,
    compare_rank: int = 16,
    planner_ff: int = 128,
    memory_slots: int = 8,
) -> FOGReasonerConfig:
    retained_memory = min(latent_slots * reasoning_steps, memory_slots)
    max_seq_len = max(64, task.prompt_length + retained_memory + 2)
    return FOGReasonerConfig(
        vocab_size=task.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
        initializer_range=0.02,
        latent_slots=latent_slots,
        reasoning_steps=reasoning_steps,
        compare_rank=compare_rank,
        planner_ff=planner_ff,
        memory_slots=memory_slots,
        n_reasoning_modes=4,
        # Keep the three objectives exactly matched at token CE.
        diversity_weight=0.0,
        route_entropy_weight=0.0,
    )


@torch.no_grad()
def initialize_by_parameter_name(model: nn.Module, *, seed: int, std: float) -> None:
    """Initialize parameters from ``(seed, full parameter name)``.

    Adding FOG-only modules cannot shift the RNG stream of shared modules.  A
    parameter such as ``backbone.blocks.0.attn.in_proj_weight`` therefore starts
    bit-identically in all three variants, independent of construction order.
    """

    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if name.endswith("bias"):
            parameter.zero_()
        elif parameter.ndim == 1:
            # RMSNorm gains use their neutral identity initialization.
            parameter.fill_(1.0)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_int("matched-name-init-v1", seed, name) % (2**63 - 1)
            )
            values = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).mul_(std)
            parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))


def build_model(
    variant: Variant,
    cfg: FOGReasonerConfig,
    *,
    model_seed: int,
) -> nn.Module:
    if variant == "direct":
        model: nn.Module = DirectLookupTransformer(cfg)
    elif variant in FOG_VARIANTS:
        model = FOGLatentReasoner(cfg)
    else:
        raise ValueError(f"unknown variant: {variant}")
    initialize_by_parameter_name(model, seed=model_seed, std=cfg.initializer_range)
    return model


def _tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda row: row[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def shared_initialization_digest(model: nn.Module) -> str:
    """Digest the complete lexical stack of one arm (not cross-arm comparable)."""

    prefixes = ("token.", "backbone.", "lm_head.")
    return _tensor_digest(
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if name.startswith(prefixes)
    )


def model_initialization_digest(model: nn.Module) -> str:
    return _tensor_digest(model.state_dict().items())


def common_initialization_report(models: dict[Variant, nn.Module]) -> dict:
    """Verify paired initialization over the canonical tensor intersection.

    Arms may have different layer counts and FFN widths.  Only lexical tensors
    that exist with identical names *and shapes* in every selected arm are
    compared.  This still gates embeddings, positions, kind embeddings,
    compatible block-0 attention/norm tensors, and the output norm/head.
    """

    if not models:
        raise ValueError("at least one model is required")
    states = {variant: model.state_dict() for variant, model in models.items()}
    variants = tuple(states)
    shared_names = set.intersection(*(set(state) for state in states.values()))
    prefixes = ("token.", "backbone.", "lm_head.")
    canonical_names = sorted(
        name
        for name in shared_names
        if name.startswith(prefixes)
        and len({tuple(states[variant][name].shape) for variant in variants}) == 1
    )
    if not canonical_names:
        raise AssertionError("no shape-compatible lexical tensors across arms")
    reference = variants[0]
    mismatched = []
    for name in canonical_names:
        reference_tensor = states[reference][name]
        if any(
            not torch.equal(reference_tensor, states[variant][name])
            for variant in variants[1:]
        ):
            mismatched.append(name)
    if mismatched:
        raise AssertionError(
            "name-stable initialization mismatch: " + ", ".join(mismatched)
        )

    # These tensors must remain paired even when direct has an extra layer and
    # a wider FFN.  The attention projections in block 0 have d_model geometry.
    required = {
        "token.weight",
        "lm_head.weight",
        "backbone.pos.weight",
        "backbone.kind.weight",
        "backbone.blocks.0.attn.in_proj_weight",
        "backbone.blocks.0.attn.out_proj.weight",
        "backbone.out_norm.weight",
    }
    missing_required = sorted(required - set(canonical_names))
    if missing_required:
        raise AssertionError(
            "canonical paired tensors are missing: " + ", ".join(missing_required)
        )
    digest = _tensor_digest(
        (name, states[reference][name]) for name in canonical_names
    )
    return {
        "scheme": "sha256(seed, full_parameter_name)_v1",
        "variants": list(variants),
        "canonical_common_state_sha256": digest,
        "canonical_tensor_count": len(canonical_names),
        "canonical_parameter_entries": sum(
            states[reference][name].numel() for name in canonical_names
        ),
        "canonical_tensor_names": canonical_names,
        "excluded_shape_mismatch_names": sorted(
            name
            for name in shared_names
            if name.startswith(prefixes) and name not in canonical_names
        ),
        "exact_match": True,
    }


def _fog_logits(
    model: FOGLatentReasoner,
    variant: Variant,
    prompt_ids: torch.Tensor,
    *,
    intervention: Intervention,
) -> torch.Tensor:
    memory, _ = model.reason(prompt_ids, return_diagnostics=False)
    if memory is None:
        raise AssertionError("FOG condition must have positive reasoning depth")
    if intervention == "zero":
        memory = torch.zeros_like(memory)
    elif intervention == "shuffle":
        memory = memory.roll(1, dims=0)
    elif intervention != "normal":
        raise ValueError(f"unknown intervention: {intervention}")

    lexical_prompt = (
        prompt_ids
        if variant == "fog_full"
        else torch.full(
            (prompt_ids.size(0), 1),
            TASK,
            dtype=torch.long,
            device=prompt_ids.device,
        )
    )
    decoder = torch.full(
        (prompt_ids.size(0), 1),
        ANSWER_BOS,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    return model.decode(lexical_prompt, memory, decoder)[:, 0]


def model_logits(
    model: nn.Module,
    variant: Variant,
    prompt_ids: torch.Tensor,
    *,
    intervention: Intervention = "normal",
) -> torch.Tensor:
    if variant == "direct":
        if intervention != "normal":
            raise ValueError("memory interventions apply only to FOG variants")
        if not isinstance(model, DirectLookupTransformer):
            raise TypeError("direct variant requires DirectLookupTransformer")
        return model.logits(prompt_ids)
    if not isinstance(model, FOGLatentReasoner):
        raise TypeError("FOG variant requires FOGLatentReasoner")
    return _fog_logits(model, variant, prompt_ids, intervention=intervention)


def _update_batch_digest(digest: "hashlib._Hash", batch: LookupBatch) -> None:
    digest.update(batch.prompt_ids.cpu().contiguous().numpy().tobytes())
    digest.update(batch.target_ids.cpu().contiguous().numpy().tobytes())


def verify_mapping_holdout(
    task: LookupTaskConfig,
    *,
    data_seed: int,
    train_examples: int,
    validation_examples: int,
    test_examples: int,
) -> dict:
    counts = {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
    }
    mapping_sets: dict[str, set[tuple[int, ...]]] = {}
    for split, count in counts.items():
        mapping_sets[split] = {
            make_example(
                task,
                data_seed=data_seed,
                split=split,  # type: ignore[arg-type]
                sample_index=index,
            ).mapping
            for index in range(count)
        }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlap = {f"{a}_{b}": len(mapping_sets[a] & mapping_sets[b]) for a, b in pairs}
    if any(overlap.values()):
        raise AssertionError("mapping/operator leakage across splits")
    return {
        "partition_key": "mapping permutation only",
        "bucket_rule": "blake2b(mapping) % 10: train=0..7, validation=8, test=9",
        "checked_examples": counts,
        "unique_mappings": {key: len(value) for key, value in mapping_sets.items()},
        "overlap": overlap,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    variant: Variant,
    task: LookupTaskConfig,
    *,
    data_seed: int,
    eval_examples: int,
    eval_batch_size: int,
    device: torch.device,
    split: Literal["validation", "test"] = "validation",
    intervention: Intervention = "normal",
) -> dict:
    if eval_examples <= 0 or eval_batch_size <= 1:
        raise ValueError("eval_examples must be positive and eval_batch_size must be > 1")
    was_training = model.training
    model.eval()
    correct = 0
    total_nll = 0.0
    digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        for start in range(0, eval_examples, eval_batch_size):
            count = min(eval_batch_size, eval_examples - start)
            batch = make_batch(
                task,
                data_seed=data_seed,
                split=split,
                start_index=start,
                batch_size=count,
            )
            _update_batch_digest(digest, batch)
            batch = batch.to(device)
            logits = model_logits(
                model,
                variant,
                batch.prompt_ids,
                intervention=intervention,
            )
            correct += int(logits.argmax(dim=-1).eq(batch.target_ids).sum())
            total_nll += float(
                F.cross_entropy(logits.float(), batch.target_ids, reduction="sum")
            )
    finally:
        model.train(was_training)
    return {
        "intervention": intervention,
        "split": split,
        "accuracy": correct / eval_examples,
        "correct": correct,
        "count": eval_examples,
        "nll": total_nll / eval_examples,
        "stream_sha256": digest.hexdigest(),
        "seconds": time.perf_counter() - started,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _json_ready_args(values: dict) -> dict:
    return {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in values.items()
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def train_variant(
    variant: Variant,
    cfg: FOGReasonerConfig,
    task: LookupTaskConfig,
    *,
    model_seed: int,
    data_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_examples: int,
    eval_batch_size: int,
    log_every: int,
    output_dir: Path,
    device: torch.device,
    evaluation_split: Literal["validation", "test"] = "validation",
    initialization_pairing: dict | None = None,
) -> dict:
    if steps <= 0 or batch_size <= 0 or log_every <= 0:
        raise ValueError("steps, batch_size, and log_every must be positive")
    model = build_model(variant, cfg, model_seed=model_seed).to(device)
    initial_shared_digest = shared_initialization_digest(model)
    initial_model_digest = model_initialization_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    trace: list[dict] = []
    stream_digest = hashlib.sha256()
    train_correct = 0
    train_count = 0
    started = time.perf_counter()
    for step in range(steps):
        batch = make_batch(
            task,
            data_seed=data_seed,
            split="train",
            start_index=step * batch_size,
            batch_size=batch_size,
        )
        _update_batch_digest(stream_digest, batch)
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model_logits(model, variant, batch.prompt_ids)
        loss = F.cross_entropy(logits.float(), batch.target_ids)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_correct = int(logits.detach().argmax(dim=-1).eq(batch.target_ids).sum())
        train_correct += batch_correct
        train_count += batch_size

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "batch_accuracy": batch_correct / batch_size,
                "grad_norm": float(grad_norm),
            }
            trace.append(row)
            print(
                f"[seed={model_seed} {variant:10s}] "
                f"step={step + 1:5d}/{steps} loss={row['loss']:.4f} "
                f"batch={100 * row['batch_accuracy']:.1f}%",
                flush=True,
            )

    train_seconds = time.perf_counter() - started
    evaluations = {
        "normal": evaluate_model(
            model,
            variant,
            task,
            data_seed=data_seed,
            eval_examples=eval_examples,
            eval_batch_size=eval_batch_size,
            device=device,
            split=evaluation_split,
            intervention="normal",
        )
    }
    if variant in FOG_VARIANTS:
        for intervention in ("zero", "shuffle"):
            evaluations[intervention] = evaluate_model(
                model,
                variant,
                task,
                data_seed=data_seed,
                eval_examples=eval_examples,
                eval_batch_size=eval_batch_size,
                device=device,
                split=evaluation_split,
                intervention=intervention,
            )
        expected_stream = evaluations["normal"]["stream_sha256"]
        if any(row["stream_sha256"] != expected_stream for row in evaluations.values()):
            raise AssertionError("FOG interventions did not receive identical eval data")

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "variant": variant,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "steps": steps,
        "batch_size": batch_size,
        "train_examples": train_count,
        "train_online_accuracy": train_correct / train_count,
        "train_seconds": train_seconds,
        "training_stream_sha256": stream_digest.hexdigest(),
        "initialization": {
            "scheme": "sha256(seed, full_parameter_name)_v1",
            "arm_lexical_state_sha256": initial_shared_digest,
            "whole_model_sha256": initial_model_digest,
            "canonical_cross_arm": initialization_pairing,
        },
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "shared_lexical_backbone": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.startswith(("token.", "backbone.", "lm_head."))
            ),
        },
        "trace": trace,
        "eval": evaluations,
        "evaluation_split": evaluation_split,
    }

    run_dir = output_dir / f"seed_{model_seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{variant}.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "variant": variant,
            "model_class": type(model).__name__,
            "model_config": asdict(cfg),
            "task_config": asdict(task),
            "model_state_dict": _cpu_state_dict(model),
            "training": {
                "model_seed": model_seed,
                "data_seed": data_seed,
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    _write_json(run_dir / f"{variant}.metrics.json", metrics)
    return metrics


def run_experiment(args: argparse.Namespace) -> dict:
    task = LookupTaskConfig(
        table_size=args.table_size,
        separate_key_value_tokens=args.separate_key_value_tokens,
    )
    task.validate()
    fog_full_steps = (
        args.reasoning_steps
        if args.fog_full_steps is None
        else args.fog_full_steps
    )
    fog_strict_steps = (
        args.reasoning_steps
        if args.fog_strict_steps is None
        else args.fog_strict_steps
    )
    if fog_full_steps <= 0 or fog_strict_steps <= 0:
        raise ValueError("FOG reasoning depths must be positive")
    # Size the shared positional table for the deepest selected condition.  The
    # reasoning-depth field itself has no parameters, so replacing it below
    # leaves every name-matched initial tensor bit-identical.
    direct_layers = args.n_layers if args.direct_layers is None else args.direct_layers
    fog_layers = args.n_layers if args.fog_layers is None else args.fog_layers
    direct_ff = args.d_ff if args.direct_ff is None else args.direct_ff
    fog_ff = args.d_ff if args.fog_ff is None else args.fog_ff
    cfg = model_config(
        task,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=fog_layers,
        d_ff=fog_ff,
        latent_slots=args.latent_slots,
        reasoning_steps=max(fog_full_steps, fog_strict_steps),
        compare_rank=args.compare_rank,
        planner_ff=args.planner_ff,
        memory_slots=args.memory_slots,
    )
    cfg.validate()
    condition_configs = {
        "direct": replace(cfg, n_layers=direct_layers, d_ff=direct_ff),
        "fog_full": replace(cfg, reasoning_steps=fog_full_steps),
        "fog_strict": replace(cfg, reasoning_steps=fog_strict_steps),
    }
    for condition_cfg in condition_configs.values():
        condition_cfg.validate()
    shared_geometry = {
        (condition_cfg.vocab_size, condition_cfg.d_model, condition_cfg.n_heads,
         condition_cfg.max_seq_len)
        for condition_cfg in condition_configs.values()
    }
    if len(shared_geometry) != 1:
        raise AssertionError("arms must share vocab/d_model/n_heads/max_seq_len")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_eval_examples = (
        args.validation_examples
        if args.evaluation_split == "validation"
        else args.test_examples
    )
    holdout_check = verify_mapping_holdout(
        task,
        data_seed=args.data_seed,
        train_examples=min(args.steps * args.batch_size, args.protocol_check_examples),
        validation_examples=min(args.validation_examples, args.protocol_check_examples),
        test_examples=min(args.test_examples, args.protocol_check_examples),
    )
    results: dict[str, dict[str, dict]] = {}
    initialization_pairing: dict[str, dict] = {}
    for model_seed in args.seeds:
        initial_models = {
            variant: build_model(
                variant, condition_configs[variant], model_seed=model_seed
            )
            for variant in args.variants
        }
        pairing = common_initialization_report(initial_models)
        initialization_pairing[str(model_seed)] = pairing
        del initial_models
        seed_results: dict[str, dict] = {}
        for variant in args.variants:
            seed_results[variant] = train_variant(
                variant,
                condition_configs[variant],
                task,
                model_seed=model_seed,
                data_seed=args.data_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                eval_examples=selected_eval_examples,
                eval_batch_size=args.eval_batch_size,
                log_every=args.log_every,
                output_dir=output_dir,
                device=device,
                evaluation_split=args.evaluation_split,
                initialization_pairing=pairing,
            )
        results[str(model_seed)] = seed_results

    training_streams = {
        metrics["training_stream_sha256"]
        for seed_results in results.values()
        for metrics in seed_results.values()
    }
    eval_streams = {
        metrics["eval"]["normal"]["stream_sha256"]
        for seed_results in results.values()
        for metrics in seed_results.values()
    }
    if len(training_streams) != 1 or len(eval_streams) != 1:
        raise AssertionError("conditions or seeds did not receive shared data streams")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "task": {
            **asdict(task),
            "vocab_size": task.vocab_size,
            "prompt_length": task.prompt_length,
            "chance_accuracy": task.chance_accuracy,
            "token_space": (
                "separate_key_value"
                if task.separate_key_value_tokens
                else "shared_state"
            ),
            "serialization": "TASK (ROW KEY MAPS_TO VALUE ROW_END)* TABLE_END QUERY KEY",
            "split_rule": (
                "blake2b(mapping) % 10: train=0..7, validation=8, test=9; "
                "mapping operators cannot cross splits"
            ),
        },
        "model_config": asdict(cfg),
        "condition_model_configs": {
            variant: asdict(condition_configs[variant])
            for variant in args.variants
        },
        "arguments": _json_ready_args(vars(args)),
        "initialization_pairing": initialization_pairing,
        "protocol_check": holdout_check,
        "shared_streams": {
            "training_sha256": next(iter(training_streams)),
            f"{args.evaluation_split}_sha256": next(iter(eval_streams)),
        },
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/matched_lookup"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-seed", type=int, default=202_608_12)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
        help="use validation during tuning; select test only for the final locked run",
    )
    parser.add_argument("--validation-examples", type=int, default=1_024)
    parser.add_argument("--test-examples", type=int, default=1_024)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--protocol-check-examples", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--table-size", type=int, default=8)
    parser.add_argument(
        "--separate-key-value-tokens",
        action="store_true",
        help=(
            "legacy calibration: allocate disjoint token IDs for keys and values; "
            "default uses one shared state-token vocabulary"
        ),
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument(
        "--direct-layers",
        type=int,
        default=None,
        help="direct-only layer count (defaults to --n-layers)",
    )
    parser.add_argument(
        "--direct-ff",
        type=int,
        default=None,
        help="direct-only FFN width (defaults to --d-ff)",
    )
    parser.add_argument(
        "--fog-layers",
        type=int,
        default=None,
        help="FOG-only layer count (defaults to --n-layers)",
    )
    parser.add_argument(
        "--fog-ff",
        type=int,
        default=None,
        help="FOG-only FFN width (defaults to --d-ff)",
    )
    parser.add_argument("--latent-slots", type=int, default=4)
    parser.add_argument("--reasoning-steps", type=int, default=2)
    parser.add_argument(
        "--fog-full-steps",
        type=int,
        default=None,
        help="override latent depth only for fog_full (for example R=1)",
    )
    parser.add_argument(
        "--fog-strict-steps",
        type=int,
        default=None,
        help="override latent depth only for fog_strict (for example R=4)",
    )
    parser.add_argument("--compare-rank", type=int, default=16)
    parser.add_argument("--planner-ff", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() > 1:
        torch.set_num_interop_threads(1)
    summary = run_experiment(args)
    compact = {
        seed: {
            variant: {
                "normal_accuracy": round(100 * metrics["eval"]["normal"]["accuracy"], 3),
                **(
                    {
                        "zero_accuracy": round(100 * metrics["eval"]["zero"]["accuracy"], 3),
                        "shuffle_accuracy": round(
                            100 * metrics["eval"]["shuffle"]["accuracy"], 3
                        ),
                    }
                    if variant in FOG_VARIANTS
                    else {}
                ),
            }
            for variant, metrics in seed_results.items()
        }
        for seed, seed_results in summary["results"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
