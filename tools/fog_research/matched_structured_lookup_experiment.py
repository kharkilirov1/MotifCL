"""Matched structured permutation-lookup gate for the FOG latent interface.

This experiment removes text serialization from associative lookup.  For an
eight-state permutation ``p``, each table row is exactly one continuous vector

    key_embedding(source) + value_embedding(p[source]) + row_type

and the final query is exactly

    key_embedding(query) + query_type.

The arms receive the same online examples, share name-stable initial
values for every common tensor, and use the same ``TinyDecoderBackbone`` and
eight-way state classifier:

* ``direct`` classifies the hidden state at the final query position;
* ``direct_bos`` appends the same learned answer-BOS interface used by FOG and
  classifies that position, but has no planner or memory;
* ``direct_hidden_memory`` encodes the prompt once, retains every prompt hidden
  state losslessly, and feeds those states as memory to a second strict-style
  pass ending in answer-BOS; it has no planner or compressor;
* ``fog_full`` recurrently constructs memory, then decodes from prompt +
  memory + a learned answer-BOS vector;
* ``fog_strict`` constructs the same memory, but the decoder sees only a
  learned neutral vector + memory + answer-BOS.

Mappings (not examples) are hash-partitioned into train/validation/test, so a
permutation cannot leak across splits under a different row order or query.
FOG evaluation includes zero memory and a deterministic target-deranged
permutation of memory between batch elements.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from fog_lmw.checkpoint import atomic_torch_save
from fog_lmw.core import TinyDecoderBackbone
from fog_lmw.memory import PersistentLatentMemory
from fog_lmw.planner import LatentPlanner


Variant = Literal[
    "direct",
    "direct_bos",
    "direct_hidden_memory",
    "fog_full",
    "fog_strict",
]
Split = Literal["train", "validation", "test"]
Intervention = Literal["normal", "zero", "target_deranged_shuffle"]

VARIANTS: tuple[Variant, ...] = (
    "direct",
    "direct_bos",
    "direct_hidden_memory",
    "fog_full",
    "fog_strict",
)
FOG_VARIANTS: tuple[Variant, ...] = ("fog_full", "fog_strict")
EXPERIMENT_NAME = "matched_structured_permutation_lookup_v1"


@dataclass(frozen=True)
class StructuredTaskConfig:
    table_size: int = 8

    def validate(self) -> None:
        # Below three states a 10-bucket permutation partition need not support
        # all three splits.
        if self.table_size < 3:
            raise ValueError("table_size must be >= 3")

    @property
    def prompt_length(self) -> int:
        # One vector per row plus one query vector.
        return self.table_size + 1

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.table_size


@dataclass(frozen=True)
class StructuredModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    max_seq_len: int = 32
    dropout: float = 0.0
    initializer_range: float = 0.02
    latent_slots: int = 4
    reasoning_steps: int = 2
    compare_rank: int = 16
    planner_ff: int = 128
    memory_slots: int = 8
    n_reasoning_modes: int = 4
    fixed_orthogonal_keys: bool = False

    def validate(self, task: StructuredTaskConfig) -> None:
        task.validate()
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("n_layers and d_ff must be positive")
        if self.latent_slots <= 0 or self.reasoning_steps <= 0:
            raise ValueError("latent_slots and reasoning_steps must be positive")
        if self.compare_rank <= 0 or self.planner_ff <= 0:
            raise ValueError("compare_rank and planner_ff must be positive")
        if self.memory_slots <= 0 or self.n_reasoning_modes <= 0:
            raise ValueError("memory_slots and n_reasoning_modes must be positive")
        if self.fixed_orthogonal_keys and self.d_model < task.table_size:
            raise ValueError(
                "fixed orthogonal keys require d_model >= table_size"
            )
        retained = min(self.latent_slots * self.reasoning_steps, self.memory_slots)
        longest = task.prompt_length + retained + 1
        if longest > self.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} is below required {longest}"
            )


@dataclass(frozen=True)
class StructuredExample:
    row_sources: tuple[int, ...]
    row_values: tuple[int, ...]
    query_key: int
    target_state: int
    mapping: tuple[int, ...]
    row_order: tuple[int, ...]
    sample_index: int
    split: Split

    @property
    def signature(self) -> tuple[int, ...]:
        return (
            *self.row_sources,
            *self.row_values,
            self.query_key,
            self.target_state,
        )


@dataclass(frozen=True)
class StructuredBatch:
    row_sources: torch.Tensor
    row_values: torch.Tensor
    query_keys: torch.Tensor
    targets: torch.Tensor
    sample_indices: tuple[int, ...]
    mappings: tuple[tuple[int, ...], ...]
    signatures: tuple[tuple[int, ...], ...]

    def to(self, device: torch.device) -> "StructuredBatch":
        return StructuredBatch(
            row_sources=self.row_sources.to(device),
            row_values=self.row_values.to(device),
            query_keys=self.query_keys.to(device),
            targets=self.targets.to(device),
            sample_indices=self.sample_indices,
            mappings=self.mappings,
            signatures=self.signatures,
        )


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _mapping_bytes(mapping: Iterable[int]) -> bytes:
    return json.dumps(list(mapping), separators=(",", ":")).encode("ascii")


def mapping_bucket(mapping: Iterable[int]) -> int:
    digest = hashlib.blake2b(_mapping_bytes(mapping), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 10


def mapping_in_split(mapping: Iterable[int], split: Split) -> bool:
    bucket = mapping_bucket(mapping)
    return (
        (split == "train" and bucket < 8)
        or (split == "validation" and bucket == 8)
        or (split == "test" and bucket == 9)
    )


def make_example(
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: Split,
    sample_index: int,
) -> StructuredExample:
    """Draw one deterministic sample from a mapping-disjoint split."""

    task.validate()
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")
    for attempt in range(10_000):
        rng = random.Random(
            _stable_int(
                EXPERIMENT_NAME, "example", data_seed, split, sample_index, attempt
            )
        )
        mapping = list(range(task.table_size))
        row_order = list(range(task.table_size))
        rng.shuffle(mapping)
        rng.shuffle(row_order)
        query = rng.randrange(task.table_size)
        if not mapping_in_split(mapping, split):
            continue
        return StructuredExample(
            row_sources=tuple(row_order),
            row_values=tuple(mapping[source] for source in row_order),
            query_key=query,
            target_state=mapping[query],
            mapping=tuple(mapping),
            row_order=tuple(row_order),
            sample_index=sample_index,
            split=split,
        )
    raise RuntimeError("could not draw an example in the requested hash split")


def make_batch(
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: Split,
    start_index: int,
    batch_size: int,
) -> StructuredBatch:
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
    return StructuredBatch(
        row_sources=torch.tensor([row.row_sources for row in examples], dtype=torch.long),
        row_values=torch.tensor([row.row_values for row in examples], dtype=torch.long),
        query_keys=torch.tensor([row.query_key for row in examples], dtype=torch.long),
        targets=torch.tensor([row.target_state for row in examples], dtype=torch.long),
        sample_indices=tuple(row.sample_index for row in examples),
        mappings=tuple(row.mapping for row in examples),
        signatures=tuple(row.signature for row in examples),
    )


class StructuredInputMixin:
    """Common embedding contract used bit-identically by all conditions."""

    task: StructuredTaskConfig
    key: nn.Embedding
    value: nn.Embedding
    row_type: nn.Parameter
    query_type: nn.Parameter

    def encode_prompt(self, batch: StructuredBatch) -> torch.Tensor:
        rows = (
            self.key(batch.row_sources)
            + self.value(batch.row_values)
            + self.row_type
        )
        query = self.key(batch.query_keys).unsqueeze(1) + self.query_type
        return torch.cat([rows, query], dim=1)


class DirectStructuredLookup(nn.Module, StructuredInputMixin):
    """Ordinary causal decoder; its final position is the query vector."""

    def __init__(self, task: StructuredTaskConfig, cfg: StructuredModelConfig):
        super().__init__()
        cfg.validate(task)
        self.task = task
        self.cfg = cfg
        self.key = nn.Embedding(task.table_size, cfg.d_model)
        self.value = nn.Embedding(task.table_size, cfg.d_model)
        self.row_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.query_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.backbone = TinyDecoderBackbone(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.d_ff,
            cfg.max_seq_len,
            cfg.dropout,
        )
        self.classifier = nn.Linear(cfg.d_model, task.table_size, bias=False)

    def logits(self, batch: StructuredBatch) -> torch.Tensor:
        prompt = self.encode_prompt(batch)
        kinds = torch.zeros(
            prompt.shape[:2], dtype=torch.long, device=prompt.device
        )
        hidden = self.backbone.forward_embeds(prompt, kinds)
        return self.classifier(hidden[:, -1])


class DirectBOSStructuredLookup(nn.Module, StructuredInputMixin):
    """Ordinary causal decoder that predicts from a learned answer-BOS.

    This is the localization control for the FOG decoding interface: it sees
    the complete structured prompt followed by exactly the same named
    ``answer_bos`` parameter as the FOG arms, but contains no planner, latent
    memory, neutral token, or recurrent pass.
    """

    def __init__(self, task: StructuredTaskConfig, cfg: StructuredModelConfig):
        super().__init__()
        cfg.validate(task)
        self.task = task
        self.cfg = cfg
        self.key = nn.Embedding(task.table_size, cfg.d_model)
        self.value = nn.Embedding(task.table_size, cfg.d_model)
        self.row_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.query_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.answer_bos = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.backbone = TinyDecoderBackbone(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.d_ff,
            cfg.max_seq_len,
            cfg.dropout,
        )
        self.classifier = nn.Linear(cfg.d_model, task.table_size, bias=False)

    def logits(self, batch: StructuredBatch) -> torch.Tensor:
        prompt = self.encode_prompt(batch)
        bos = self.answer_bos.expand(prompt.size(0), 1, -1)
        joined = torch.cat([prompt, bos], dim=1)
        kinds = torch.zeros(
            joined.shape[:2], dtype=torch.long, device=joined.device
        )
        hidden = self.backbone.forward_embeds(joined, kinds)
        return self.classifier(hidden[:, -1])


class DirectHiddenMemoryStructuredLookup(nn.Module, StructuredInputMixin):
    """Two-pass lossless-memory control for the strict FOG interface.

    The first pass encodes the complete prompt.  Every resulting prompt hidden
    state is retained without selection, compression, or a learned memory
    update.  The second pass receives ``neutral + hidden_memory + answer_bos``
    with the same lexical/memory kind layout as strict FOG and predicts only at
    answer-BOS.  Thus prompt content can reach the answer exclusively through
    the retained hidden states.
    """

    def __init__(self, task: StructuredTaskConfig, cfg: StructuredModelConfig):
        super().__init__()
        cfg.validate(task)
        self.task = task
        self.cfg = cfg
        self.key = nn.Embedding(task.table_size, cfg.d_model)
        self.value = nn.Embedding(task.table_size, cfg.d_model)
        self.row_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.query_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.neutral = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.answer_bos = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.backbone = TinyDecoderBackbone(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.d_ff,
            cfg.max_seq_len,
            cfg.dropout,
        )
        self.classifier = nn.Linear(cfg.d_model, task.table_size, bias=False)

    def encode_hidden_memory(self, batch: StructuredBatch) -> torch.Tensor:
        prompt = self.encode_prompt(batch)
        kinds = torch.zeros(
            prompt.shape[:2], dtype=torch.long, device=prompt.device
        )
        return self.backbone.forward_embeds(prompt, kinds)

    def decode_hidden_memory(self, hidden_memory: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_memory.size(0)
        neutral = self.neutral.expand(batch_size, 1, -1)
        bos = self.answer_bos.expand(batch_size, 1, -1)
        joined = torch.cat([neutral, hidden_memory, bos], dim=1)
        kinds = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=joined.device),
                torch.ones(
                    batch_size,
                    hidden_memory.size(1),
                    dtype=torch.long,
                    device=joined.device,
                ),
                torch.zeros(batch_size, 1, dtype=torch.long, device=joined.device),
            ],
            dim=1,
        )
        hidden = self.backbone.forward_embeds(joined, kinds)
        return self.classifier(hidden[:, -1])

    def logits(self, batch: StructuredBatch) -> torch.Tensor:
        return self.decode_hidden_memory(self.encode_hidden_memory(batch))


class FOGStructuredLookup(nn.Module, StructuredInputMixin):
    """FOG reason loop operating directly on structured continuous vectors."""

    def __init__(self, task: StructuredTaskConfig, cfg: StructuredModelConfig):
        super().__init__()
        cfg.validate(task)
        self.task = task
        self.cfg = cfg
        self.key = nn.Embedding(task.table_size, cfg.d_model)
        self.value = nn.Embedding(task.table_size, cfg.d_model)
        self.row_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.query_type = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.neutral = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.answer_bos = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.backbone = TinyDecoderBackbone(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.d_ff,
            cfg.max_seq_len,
            cfg.dropout,
        )
        self.classifier = nn.Linear(cfg.d_model, task.table_size, bias=False)
        self.planner = LatentPlanner(
            d_model=cfg.d_model,
            latent_slots=cfg.latent_slots,
            compare_rank=cfg.compare_rank,
            planner_ff=cfg.planner_ff,
            n_reasoning_modes=cfg.n_reasoning_modes,
        )
        self.memory = PersistentLatentMemory(
            cfg.d_model, cfg.compare_rank, cfg.memory_slots
        )

    def _backbone_with_memory(
        self, prompt: torch.Tensor, memory: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, prompt_length, _ = prompt.shape
        if memory is None:
            joined = prompt
            kinds = torch.zeros(
                batch_size, prompt_length, dtype=torch.long, device=prompt.device
            )
        else:
            joined = torch.cat([prompt, memory], dim=1)
            kinds = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        prompt_length,
                        dtype=torch.long,
                        device=prompt.device,
                    ),
                    torch.ones(
                        batch_size,
                        memory.size(1),
                        dtype=torch.long,
                        device=prompt.device,
                    ),
                ],
                dim=1,
            )
        hidden = self.backbone.forward_embeds(joined, kinds)
        mask = torch.ones(
            hidden.shape[:2], dtype=torch.bool, device=hidden.device
        )
        return hidden, mask

    def reason_embeds(self, prompt: torch.Tensor) -> torch.Tensor:
        """Build memory without converting structured rows into token IDs."""

        memory: torch.Tensor | None = None
        for _ in range(self.cfg.reasoning_steps):
            hidden, context_mask = self._backbone_with_memory(prompt, memory)
            latent, _ = self.planner(hidden, context_mask=context_mask)
            memory, _ = self.memory(memory, latent)
        if memory is None:
            raise AssertionError("positive reasoning_steps must construct memory")
        return memory

    def decode_embeds(
        self,
        variant: Variant,
        prompt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prompt.size(0)
        if variant == "fog_full":
            lexical = prompt
        elif variant == "fog_strict":
            lexical = self.neutral.expand(batch_size, 1, -1)
        else:
            raise ValueError(f"FOG decoder cannot run variant {variant}")
        bos = self.answer_bos.expand(batch_size, 1, -1)
        joined = torch.cat([lexical, memory, bos], dim=1)
        kinds = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    lexical.size(1),
                    dtype=torch.long,
                    device=prompt.device,
                ),
                torch.ones(
                    batch_size,
                    memory.size(1),
                    dtype=torch.long,
                    device=prompt.device,
                ),
                torch.zeros(batch_size, 1, dtype=torch.long, device=prompt.device),
            ],
            dim=1,
        )
        hidden = self.backbone.forward_embeds(joined, kinds)
        return self.classifier(hidden[:, -1])

    def logits(
        self,
        variant: Variant,
        batch: StructuredBatch,
        *,
        intervention: Intervention = "normal",
    ) -> torch.Tensor:
        prompt = self.encode_prompt(batch)
        memory = self.reason_embeds(prompt)
        if intervention == "zero":
            memory = torch.zeros_like(memory)
        elif intervention == "target_deranged_shuffle":
            donors = target_deranged_indices(batch.targets)
            memory = memory.index_select(0, donors)
        elif intervention != "normal":
            raise ValueError(f"unknown intervention: {intervention}")
        return self.decode_embeds(variant, prompt, memory)


@torch.no_grad()
def initialize_by_parameter_name(model: nn.Module, *, seed: int, std: float) -> None:
    """Initialize each parameter from ``(seed, full name)`` independently."""

    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if name.endswith("bias"):
            parameter.zero_()
        elif parameter.ndim == 1:
            parameter.fill_(1.0)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_int(EXPERIMENT_NAME, "name-init", seed, name) % (2**63 - 1)
            )
            values = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ).mul_(std)
            parameter.copy_(values.to(parameter.device, parameter.dtype))


def build_model(
    variant: Variant,
    task: StructuredTaskConfig,
    cfg: StructuredModelConfig,
    *,
    model_seed: int,
) -> nn.Module:
    if variant == "direct":
        model: nn.Module = DirectStructuredLookup(task, cfg)
    elif variant == "direct_bos":
        model = DirectBOSStructuredLookup(task, cfg)
    elif variant == "direct_hidden_memory":
        model = DirectHiddenMemoryStructuredLookup(task, cfg)
    elif variant in FOG_VARIANTS:
        model = FOGStructuredLookup(task, cfg)
    else:
        raise ValueError(f"unknown variant: {variant}")
    initialize_by_parameter_name(model, seed=model_seed, std=cfg.initializer_range)
    if cfg.fixed_orthogonal_keys:
        # A frozen, shared one-hot key basis is an explicit calibration control:
        # it removes the need to discover equality geometry while leaving value
        # routing, row selection, and the complete backbone trainable.
        if not isinstance(model, StructuredInputMixin):
            raise TypeError("structured model must implement StructuredInputMixin")
        with torch.no_grad():
            model.key.weight.zero_()
            model.key.weight[:, : task.table_size].copy_(
                torch.eye(task.table_size, dtype=model.key.weight.dtype)
            )
        model.key.weight.requires_grad_(False)
    return model


def model_logits(
    model: nn.Module,
    variant: Variant,
    batch: StructuredBatch,
    *,
    intervention: Intervention = "normal",
) -> torch.Tensor:
    if variant in ("direct", "direct_bos", "direct_hidden_memory"):
        if intervention != "normal":
            raise ValueError("memory interventions apply only to FOG variants")
        expected_type = (
            DirectStructuredLookup
            if variant == "direct"
            else (
                DirectBOSStructuredLookup
                if variant == "direct_bos"
                else DirectHiddenMemoryStructuredLookup
            )
        )
        if not isinstance(model, expected_type):
            raise TypeError(f"{variant} variant requires {expected_type.__name__}")
        return model.logits(batch)
    if not isinstance(model, FOGStructuredLookup):
        raise TypeError("FOG variants require FOGStructuredLookup")
    return model.logits(variant, batch, intervention=intervention)


def target_deranged_indices(targets: torch.Tensor) -> torch.Tensor:
    """Return a true donor permutation with a different target for every row.

    A deterministic bipartite matching is used instead of ``roll(1)`` because
    adjacent examples can share the same answer.  The result is a permutation:
    every memory is used exactly once and no example receives memory whose
    source target equals its own target.
    """

    if targets.ndim != 1 or targets.numel() < 2:
        raise ValueError("target-deranged shuffle requires a 1D batch of size >= 2")
    values = [int(value) for value in targets.detach().cpu().tolist()]
    count = len(values)
    # Process the most constrained (most frequent) target classes first.
    frequencies = {value: values.count(value) for value in set(values)}
    receivers = sorted(range(count), key=lambda i: (-frequencies[values[i]], i))
    donor_owner = [-1] * count

    def assign(receiver: int, seen: set[int]) -> bool:
        for donor in range(count):
            if donor in seen or values[donor] == values[receiver]:
                continue
            seen.add(donor)
            previous = donor_owner[donor]
            if previous < 0 or assign(previous, seen):
                donor_owner[donor] = receiver
                return True
        return False

    for receiver in receivers:
        if not assign(receiver, set()):
            raise ValueError(
                "batch has no target-deranged memory permutation; increase the "
                "evaluation batch size or use a more balanced batch"
            )
    donors = [-1] * count
    for donor, receiver in enumerate(donor_owner):
        donors[receiver] = donor
    result = torch.tensor(donors, dtype=torch.long, device=targets.device)
    if sorted(donors) != list(range(count)):
        raise AssertionError("target derangement is not a permutation")
    if torch.any(targets.index_select(0, result).eq(targets)):
        raise AssertionError("target derangement paired an equal target")
    return result


def _tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda row: row[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def common_initialization_report(models: dict[Variant, nn.Module]) -> dict:
    if not models:
        raise ValueError("at least one model is required")
    states = {variant: model.state_dict() for variant, model in models.items()}
    variants = tuple(states)
    common_names = sorted(set.intersection(*(set(state) for state in states.values())))
    required_prefixes = (
        "key.",
        "value.",
        "row_type",
        "query_type",
        "answer_bos",
        "neutral",
        "backbone.",
        "classifier.",
    )
    shared_names = [
        name
        for name in common_names
        if name.startswith(required_prefixes)
        and len({tuple(states[variant][name].shape) for variant in variants}) == 1
    ]
    mismatched = [
        name
        for name in shared_names
        if any(
            not torch.equal(states[variants[0]][name], states[variant][name])
            for variant in variants[1:]
        )
    ]
    if mismatched:
        raise AssertionError("name-stable initialization mismatch: " + ", ".join(mismatched))
    required = {
        "key.weight",
        "value.weight",
        "row_type",
        "query_type",
        "backbone.blocks.0.attn.in_proj_weight",
        "backbone.out_norm.weight",
        "classifier.weight",
    }
    missing = sorted(required - set(shared_names))
    if missing:
        raise AssertionError("required paired tensors are missing: " + ", ".join(missing))
    reference = variants[0]
    return {
        "scheme": "sha256(seed, full_parameter_name)_v1",
        "variants": list(variants),
        "exact_match": True,
        "shared_state_sha256": _tensor_digest(
            (name, states[reference][name]) for name in shared_names
        ),
        "shared_tensor_count": len(shared_names),
        "shared_parameter_entries": sum(
            states[reference][name].numel() for name in shared_names
        ),
        "shared_tensor_names": shared_names,
    }


def parameter_report(models: dict[Variant, nn.Module]) -> dict:
    report = {}
    for variant, model in models.items():
        named = dict(model.named_parameters())
        total = sum(parameter.numel() for parameter in named.values())
        fog_only = sum(
            parameter.numel()
            for name, parameter in named.items()
            if name.startswith(("planner.", "memory.", "neutral"))
        )
        answer_bos = sum(
            parameter.numel()
            for name, parameter in named.items()
            if name.startswith("answer_bos")
        )
        architecture_only = fog_only + answer_bos
        report[variant] = {
            "total": total,
            "trainable": sum(
                parameter.numel()
                for parameter in named.values()
                if parameter.requires_grad
            ),
            "shared_structured_stack": total - architecture_only,
            # Preserve the historical aggregate while exposing the BOS control
            # separately for direct_bos/FOG interface localization.
            "fog_only": architecture_only if variant in FOG_VARIANTS else 0,
            "answer_bos": answer_bos,
            "forward_active": total,
        }
    return report


def verify_mapping_holdout(
    task: StructuredTaskConfig,
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
    mappings: dict[str, set[tuple[int, ...]]] = {}
    for split, count in counts.items():
        mappings[split] = {
            make_example(
                task,
                data_seed=data_seed,
                split=split,  # type: ignore[arg-type]
                sample_index=index,
            ).mapping
            for index in range(count)
        }
    overlaps = {
        "train_validation": len(mappings["train"] & mappings["validation"]),
        "train_test": len(mappings["train"] & mappings["test"]),
        "validation_test": len(mappings["validation"] & mappings["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError("mapping/operator leakage across splits")
    return {
        "partition_key": "mapping permutation only",
        "bucket_rule": "blake2b(mapping) % 10: train=0..7, validation=8, test=9",
        "checked_examples": counts,
        "unique_mappings": {split: len(rows) for split, rows in mappings.items()},
        "overlap": overlaps,
    }


def _update_stream_digest(digest: "hashlib._Hash", batch: StructuredBatch) -> None:
    for tensor in (
        batch.row_sources,
        batch.row_values,
        batch.query_keys,
        batch.targets,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    variant: Variant,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    eval_examples: int,
    eval_batch_size: int,
    device: torch.device,
    split: Literal["validation", "test"] = "validation",
    intervention: Intervention = "normal",
) -> dict:
    if eval_examples <= 0 or eval_batch_size < 2:
        raise ValueError("eval_examples must be positive and eval_batch_size >= 2")
    was_training = model.training
    model.eval()
    correct = 0
    total_nll = 0.0
    digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        for start in range(0, eval_examples, eval_batch_size):
            count = min(eval_batch_size, eval_examples - start)
            if intervention == "target_deranged_shuffle" and count < 2:
                raise ValueError(
                    "target-deranged evaluation requires no singleton final batch"
                )
            batch = make_batch(
                task,
                data_seed=data_seed,
                split=split,
                start_index=start,
                batch_size=count,
            )
            _update_stream_digest(digest, batch)
            batch = batch.to(device)
            logits = model_logits(model, variant, batch, intervention=intervention)
            correct += int(logits.argmax(dim=-1).eq(batch.targets).sum())
            total_nll += float(
                F.cross_entropy(logits.float(), batch.targets, reduction="sum")
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    data_seed: int,
    split: Literal["validation", "test"],
    eval_examples: int,
    eval_batch_size: int,
    device: torch.device,
    output_path: Path | None = None,
) -> dict:
    """Evaluate a saved arm without constructing an optimizer or training.

    This is the intended locked-test entry point.  Hyperparameters can be
    selected on validation, then the exact checkpoint can be evaluated once by
    passing ``split="test"``.  FOG checkpoints automatically receive normal,
    zero-memory, and target-deranged-memory evaluations on the same stream.
    """

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            f"checkpoint experiment is not {EXPERIMENT_NAME}: "
            f"{payload.get('experiment')!r}"
        )
    variant = payload.get("variant")
    if variant not in VARIANTS:
        raise ValueError(f"checkpoint has unknown variant: {variant!r}")
    task = StructuredTaskConfig(**payload["task_config"])
    cfg = StructuredModelConfig(**payload["model_config"])
    cfg.validate(task)
    model_seed = int(payload.get("metrics", {}).get("model_seed", 0))
    model = build_model(variant, task, cfg, model_seed=model_seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)

    evaluations = {
        "normal": evaluate_model(
            model,
            variant,
            task,
            data_seed=data_seed,
            eval_examples=eval_examples,
            eval_batch_size=eval_batch_size,
            device=device,
            split=split,
            intervention="normal",
        )
    }
    if variant in FOG_VARIANTS:
        for intervention in ("zero", "target_deranged_shuffle"):
            evaluations[intervention] = evaluate_model(
                model,
                variant,
                task,
                data_seed=data_seed,
                eval_examples=eval_examples,
                eval_batch_size=eval_batch_size,
                device=device,
                split=split,
                intervention=intervention,
            )
        streams = {row["stream_sha256"] for row in evaluations.values()}
        if len(streams) != 1:
            raise AssertionError("checkpoint interventions received different streams")

    result = {
        "experiment": EXPERIMENT_NAME,
        "mode": "checkpoint_only_evaluation",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _file_sha256(checkpoint_path),
            "variant": variant,
            "model_seed": model_seed,
            "training_stream_sha256": payload.get("metrics", {}).get(
                "training_stream_sha256"
            ),
        },
        "task_config": asdict(task),
        "model_config": asdict(cfg),
        "data_seed": data_seed,
        "split": split,
        "eval_examples": eval_examples,
        "eval_batch_size": eval_batch_size,
        "eval": evaluations,
    }
    destination = output_path or checkpoint_path.parent / (
        f"{checkpoint_path.stem}.{split}.evaluation.json"
    )
    _write_json(destination, result)
    result["output_path"] = str(destination)
    return result


def train_variant(
    variant: Variant,
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
    log_every: int,
    output_dir: Path,
    device: torch.device,
    evaluation_split: Literal["validation", "test"],
    initialization_pairing: dict,
    parameters: dict,
) -> dict:
    model = build_model(variant, task, cfg, model_seed=model_seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    trace = []
    train_correct = 0
    train_count = 0
    stream_digest = hashlib.sha256()
    started = time.perf_counter()
    for step in range(steps):
        batch = make_batch(
            task,
            data_seed=data_seed,
            split="train",
            start_index=step * batch_size,
            batch_size=batch_size,
        )
        _update_stream_digest(stream_digest, batch)
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model_logits(model, variant, batch)
        loss = F.cross_entropy(logits.float(), batch.targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_correct = int(logits.detach().argmax(dim=-1).eq(batch.targets).sum())
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
                f"[seed={model_seed} {variant:10s}] step={step + 1:4d}/{steps} "
                f"loss={row['loss']:.4f} batch={100 * row['batch_accuracy']:.1f}%",
                flush=True,
            )

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
        )
    }
    if variant in FOG_VARIANTS:
        for intervention in ("zero", "target_deranged_shuffle"):
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
        streams = {row["stream_sha256"] for row in evaluations.values()}
        if len(streams) != 1:
            raise AssertionError("memory interventions did not receive identical data")

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "variant": variant,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "steps": steps,
        "batch_size": batch_size,
        "train_examples": train_count,
        "train_online_accuracy": train_correct / train_count,
        "train_seconds": time.perf_counter() - started,
        "training_stream_sha256": stream_digest.hexdigest(),
        "initialization_pairing": initialization_pairing,
        "parameters": parameters,
        "trace": trace,
        "evaluation_split": evaluation_split,
        "eval": evaluations,
    }
    run_dir = output_dir / f"seed_{model_seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "format_version": 1,
            "experiment": EXPERIMENT_NAME,
            "variant": variant,
            "task_config": asdict(task),
            "model_config": asdict(cfg),
            "model_state_dict": _cpu_state_dict(model),
            "metrics": metrics,
        },
        run_dir / f"{variant}.pt",
    )
    _write_json(run_dir / f"{variant}.metrics.json", metrics)
    return metrics


def run_experiment(args: argparse.Namespace) -> dict:
    if args.steps <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("steps, batch_size, and log_every must be positive")
    task = StructuredTaskConfig(table_size=args.table_size)
    retained = min(args.latent_slots * args.reasoning_steps, args.memory_slots)
    cfg = StructuredModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=max(16, task.prompt_length + retained + 1),
        latent_slots=args.latent_slots,
        reasoning_steps=args.reasoning_steps,
        compare_rank=args.compare_rank,
        planner_ff=args.planner_ff,
        memory_slots=args.memory_slots,
        fixed_orthogonal_keys=args.fixed_orthogonal_keys,
    )
    cfg.validate(task)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_eval_examples = (
        args.validation_examples
        if args.evaluation_split == "validation"
        else args.test_examples
    )
    protocol = verify_mapping_holdout(
        task,
        data_seed=args.data_seed,
        train_examples=min(args.steps * args.batch_size, args.protocol_check_examples),
        validation_examples=min(args.validation_examples, args.protocol_check_examples),
        test_examples=min(args.test_examples, args.protocol_check_examples),
    )
    all_results: dict[str, dict[str, dict]] = {}
    initialization: dict[str, dict] = {}
    parameter_counts: dict[str, dict[str, dict]] = {}
    for seed in args.seeds:
        initial_models = {
            variant: build_model(variant, task, cfg, model_seed=seed)
            for variant in args.variants
        }
        pairing = common_initialization_report(initial_models)
        counts = parameter_report(initial_models)
        initialization[str(seed)] = pairing
        parameter_counts[str(seed)] = counts
        del initial_models
        seed_results = {}
        for variant in args.variants:
            seed_results[variant] = train_variant(
                variant,
                task,
                cfg,
                model_seed=seed,
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
                parameters=counts[variant],
            )
        all_results[str(seed)] = seed_results

    training_streams = {
        row["training_stream_sha256"]
        for seed_rows in all_results.values()
        for row in seed_rows.values()
    }
    eval_streams = {
        row["eval"]["normal"]["stream_sha256"]
        for seed_rows in all_results.values()
        for row in seed_rows.values()
    }
    if len(training_streams) != 1 or len(eval_streams) != 1:
        raise AssertionError("arms or seeds did not receive identical streams")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "task": {
            **asdict(task),
            "prompt_length": task.prompt_length,
            "chance_accuracy": task.chance_accuracy,
            "row_representation": "key(src) + value(dst) + row_type (one vector)",
            "query_representation": "key(query) + query_type (one vector)",
            "split_rule": "mapping-only blake2b bucket, 80/10/10",
        },
        "model_config": asdict(cfg),
        "arguments": {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        },
        "protocol_check": protocol,
        "initialization_pairing": initialization,
        "parameter_counts": parameter_counts,
        "shared_streams": {
            "training_sha256": next(iter(training_streams)),
            f"{args.evaluation_split}_sha256": next(iter(eval_streams)),
        },
        "results": all_results,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/matched_structured_lookup")
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="load one saved arm and evaluate it without any training",
    )
    parser.add_argument(
        "--checkpoint-eval-output",
        type=Path,
        default=None,
        help="JSON destination for --evaluate-checkpoint",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--data-seed", type=int, default=202_608_12)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--test-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--protocol-check-examples", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--table-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--latent-slots", type=int, default=4)
    parser.add_argument("--reasoning-steps", type=int, default=2)
    parser.add_argument("--compare-rank", type=int, default=16)
    parser.add_argument("--planner-ff", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument(
        "--fixed-orthogonal-keys",
        action="store_true",
        help=(
            "calibration control: freeze a shared one-hot key basis so arms do "
            "not first need to learn state-equality geometry"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() > 1:
        torch.set_num_interop_threads(1)
    if args.evaluate_checkpoint is not None:
        selected_examples = (
            args.validation_examples
            if args.evaluation_split == "validation"
            else args.test_examples
        )
        result = evaluate_checkpoint(
            args.evaluate_checkpoint,
            data_seed=args.data_seed,
            split=args.evaluation_split,
            eval_examples=selected_examples,
            eval_batch_size=args.eval_batch_size,
            device=torch.device(args.device),
            output_path=args.checkpoint_eval_output,
        )
        compact = {
            "checkpoint": result["checkpoint"],
            "split": result["split"],
            "eval": {
                name: round(100 * row["accuracy"], 3)
                for name, row in result["eval"].items()
            },
            "output_path": result["output_path"],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
        return
    summary = run_experiment(args)
    compact = {
        seed: {
            variant: {
                key: round(100 * row["accuracy"], 3)
                for key, row in metrics["eval"].items()
            }
            for variant, metrics in seed_rows.items()
        }
        for seed, seed_rows in summary["results"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
