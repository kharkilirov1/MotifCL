#!/usr/bin/env python3
"""Train the 10M FOG candidate on TinyStories and GSM8K.

The script is intentionally step based and single-process.  It supports a
streaming Hugging Face source or local TXT/JSONL files, token-normalized
gradient accumulation, BF16/FP16 autocast, atomic checkpoints, deterministic
validation splits, and complete optimizer/RNG resume.

Typical workflow::

    python train_real.py tokenizer --output tokenizer/tokenizer.json
    python train_real.py init-model --tokenizer tokenizer/tokenizer.json
    python train_real.py pretrain --tokenizer tokenizer/tokenizer.json
    python train_real.py sft --tokenizer tokenizer/tokenizer.json \
        --init-checkpoint checkpoints/pretrain/last.pt
    python train_real.py evaluate-gsm8k --tokenizer tokenizer/tokenizer.json \
        --checkpoint checkpoints/sft/best.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from itertools import islice
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any

import torch

from fog_lmw import (
    FOG_10M_PARAMETER_COUNT,
    FOG_10M_VOCAB_SIZE,
    FOG_BINDING_V2_10M_PARAMETER_COUNT,
    FOG_MACHINE_V3_10M_PARAMETER_COUNT,
    FOGReasonerConfig,
    FOGLatentReasoner,
    fog_10m_config,
    fog_binding_v2_10m_config,
    fog_machine_v3_10m_config,
)
from fog_lmw.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from fog_lmw.data import (
    BPETokenizer,
    PromptResponseCollator,
    TextBlockCollator,
    extract_gsm8k_final,
    iter_local_records,
    load_hf_dataset,
)


TINYSTORIES_ID = "roneneldan/TinyStories"
TINYSTORIES_CONFIG = "default"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
GSM8K_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        if requested in {"bf16", "fp16"} and device.type != "cuda":
            # CPU BF16 is supported on some machines, but FP32 is the portable
            # default and the least surprising choice for this reference code.
            if requested == "fp16":
                raise ValueError("fp16 training requires CUDA")
        return requested
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    if device.type == "cuda":
        return "fp16"
    return "fp32"


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def unique_parameter_count(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) not in seen:
            total += parameter.numel()
            seen.add(id(parameter))
    return total


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return checkpoint metadata without argparse callback objects."""

    cleaned: dict[str, Any] = {}
    for name, value in vars(args).items():
        if callable(value):
            continue
        if isinstance(value, Path):
            value = str(value)
        cleaned[name] = value
    return cleaned


def build_optimizer(
    model: torch.nn.Module, *, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps)")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size < 1:
        raise ValueError("batch size must be >= 1")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class CyclingBatchSource:
    """Restart a deterministic record factory and remember its exact cursor."""

    def __init__(
        self,
        record_factory: Callable[[int], Iterable[Mapping[str, Any]]],
        collator: Callable[[Sequence[Mapping[str, Any]]], dict[str, torch.Tensor]],
        batch_size: int,
        *,
        epoch: int = 0,
        batch_in_epoch: int = 0,
    ) -> None:
        self.record_factory = record_factory
        self.collator = collator
        self.batch_size = batch_size
        self.epoch = epoch
        self.batch_in_epoch = 0
        self._iterator: Iterator[list[Mapping[str, Any]]] | None = None
        self._restart(skip=batch_in_epoch)

    def _restart(self, *, skip: int = 0) -> None:
        self._iterator = iter(chunked(self.record_factory(self.epoch), self.batch_size))
        self.batch_in_epoch = 0
        for _ in range(skip):
            try:
                next(self._iterator)
            except StopIteration as exc:
                raise ValueError("saved data cursor exceeds the reconstructed epoch") from exc
            self.batch_in_epoch += 1

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while True:
            assert self._iterator is not None
            try:
                records = next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self._restart()
                continue
            self.batch_in_epoch += 1
            try:
                return self.collator(records)
            except ValueError as exc:
                # Blank/too-short text-only batches are legal data noise.  Do
                # not silently swallow schema or prompt/response errors.
                if "fewer than two usable text tokens" not in str(exc):
                    raise

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch}


def shuffled_local_factory(
    paths: Sequence[str],
    *,
    required_fields: Sequence[str],
    seed: int,
    buffer_size: int = 500,
) -> Callable[[int], Iterable[Mapping[str, Any]]]:
    # Check if empty
    try:
        next(iter_local_records(paths, required_fields=required_fields))
    except StopIteration:
        raise ValueError("local dataset is empty")

    def factory(epoch: int):
        rng = random.Random(seed + epoch)
        records = iter_local_records(paths, required_fields=required_fields)
        buffer = []
        # Fill initial buffer
        for _ in range(buffer_size):
            try:
                buffer.append(next(records))
            except StopIteration:
                break

        if not buffer:
            return

        # Stream with buffer shuffle
        for item in records:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item

        # Drain buffer
        rng.shuffle(buffer)
        yield from buffer

    return factory


def hf_factory(
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    revision: str | None,
    streaming: bool,
    seed: int,
    shuffle_buffer: int,
) -> Callable[[int], Iterable[Mapping[str, Any]]]:
    def factory(epoch: int):
        dataset = load_hf_dataset(
            dataset_id,
            config=config,
            split=split,
            revision=revision,
            streaming=streaming,
        )
        if hasattr(dataset, "shuffle"):
            options: dict[str, Any] = {"seed": seed + epoch}
            if streaming:
                options["buffer_size"] = shuffle_buffer
            dataset = dataset.shuffle(**options)
        return iter(dataset)

    return factory


def tokenizer_from_args(args: argparse.Namespace) -> BPETokenizer:
    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != FOG_10M_VOCAB_SIZE and not getattr(
        args, "allow_nonstandard_vocab", False
    ):
        raise ValueError(
            f"tokenizer has {tokenizer.vocab_size} tokens; the exact 10M preset "
            f"requires {FOG_10M_VOCAB_SIZE}. Pass --allow-nonstandard-vocab "
            "only for smoke tests."
        )
    return tokenizer


def config_from_checkpoint(path: str | Path) -> FOGReasonerConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return FOGReasonerConfig(**payload["model_config"])


def new_model_config(
    architecture: str,
    *,
    vocab_size: int,
    max_seq_len: int,
    reasoning_steps: int,
    dropout: float,
) -> FOGReasonerConfig:
    """Resolve an explicit new-model architecture without checkpoint guessing."""
    if architecture == "auto":
        architecture = "query_bound_v2"
    factory = {
        "legacy_v1": fog_10m_config,
        "query_bound_v2": fog_binding_v2_10m_config,
        "register_machine_v3": fog_machine_v3_10m_config,
    }.get(architecture)
    if factory is None:
        raise ValueError(f"unsupported architecture {architecture!r}")
    return factory(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        reasoning_steps=reasoning_steps,
        dropout=dropout,
    )


def make_model(
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> tuple[FOGLatentReasoner, FOGReasonerConfig]:
    source = getattr(args, "resume", None) or getattr(args, "init_checkpoint", None)
    if source:
        config = config_from_checkpoint(source)
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError("checkpoint vocabulary size differs from tokenizer")
        requested = getattr(args, "architecture", "auto")
        if requested != "auto" and requested != config.architecture_version:
            raise ValueError(
                f"--architecture={requested} conflicts with checkpoint "
                f"architecture {config.architecture_version}"
            )
    else:
        config = new_model_config(
            getattr(args, "architecture", "auto"),
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_seq_len,
            reasoning_steps=getattr(args, "reasoning_steps", 4),
            dropout=args.dropout,
        )
    model = FOGLatentReasoner(config).to(device)
    if getattr(args, "init_checkpoint", None):
        load_training_checkpoint(
            args.init_checkpoint,
            model=model,
            tokenizer_path=args.tokenizer,
            restore_rng=False,
        )
    return model, config


def train_tokenizer(args: argparse.Namespace) -> None:
    if args.offline_byte_fallback:
        tokenizer = BPETokenizer.byte_fallback(vocab_size=args.vocab_size)
        output = tokenizer.save(args.output)
        metadata = {
            "vocab_size": tokenizer.vocab_size,
            "kind": "byte-fallback-with-reserved-ids",
            "dataset_id": None,
            "revision": None,
            "max_samples": 0,
            "special_tokens": asdict(tokenizer.special_tokens),
        }
        output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"tokenizer": str(output), **metadata}, indent=2))
        return
    if args.local_data:
        records: Iterable[Mapping[str, Any]] = iter_local_records(
            args.local_data, text_field=args.text_field
        )
    else:
        records = load_hf_dataset(
            args.dataset_id,
            config=args.dataset_config,
            split=args.train_split,
            revision=args.revision,
            streaming=True,
        )
        if hasattr(records, "shuffle"):
            records = records.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    def texts() -> Iterator[str]:
        for index, row in enumerate(islice(records, args.max_samples)):
            text = row.get(args.text_field)
            if not isinstance(text, str):
                raise TypeError(f"record {index} field {args.text_field!r} must be str")
            if text.strip():
                yield text

    tokenizer = BPETokenizer.train(
        texts(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        length=args.max_samples,
    )
    if tokenizer.vocab_size != args.vocab_size:
        raise RuntimeError(
            f"BPE produced {tokenizer.vocab_size} tokens, expected {args.vocab_size}; "
            "increase --max-samples or reduce --min-frequency"
        )
    output = tokenizer.save(args.output)
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "dataset_id": None if args.local_data else args.dataset_id,
        "dataset_config": None if args.local_data else args.dataset_config,
        "revision": None if args.local_data else args.revision,
        "max_samples": args.max_samples,
        "special_tokens": asdict(tokenizer.special_tokens),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tokenizer": str(output), **metadata}, indent=2))


def init_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tokenizer = tokenizer_from_args(args)
    config = new_model_config(
        args.architecture,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        reasoning_steps=args.reasoning_steps,
        dropout=args.dropout,
    )
    model = FOGLatentReasoner(config)
    count = unique_parameter_count(model)
    if tokenizer.vocab_size == FOG_10M_VOCAB_SIZE and args.max_seq_len == 512:
        expected = {
            "legacy_v1": FOG_10M_PARAMETER_COUNT,
            "query_bound_v2": FOG_BINDING_V2_10M_PARAMETER_COUNT,
            "register_machine_v3": FOG_MACHINE_V3_10M_PARAMETER_COUNT,
        }[config.architecture_version]
        if count != expected:
            raise AssertionError(
                f"10M parameter contract failed for "
                f"{config.architecture_version}: {count} != {expected}"
            )
    save_training_checkpoint(
        args.output,
        model=model,
        config=config,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=0,
        consumed_tokens=0,
        tokenizer_path=args.tokenizer,
        training_args=serializable_args(args),
        extra={"stage": "initialized"},
    )
    print(json.dumps({"checkpoint": args.output, "parameters": count}, indent=2))


def finite_eval(
    model: FOGLatentReasoner,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    precision: str,
    stage: str,
    decoder_mode: str,
    bos_token_id: int,
    max_batches: int,
    reasoning_steps: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    weighted_accuracy = 0.0
    target_tokens = 0
    try:
        with torch.no_grad():
            for batch_index, raw in enumerate(batches):
                if batch_index >= max_batches:
                    break
                batch = move_batch(raw, device)
                with autocast_context(device, precision):
                    if stage == "pretrain":
                        loss, stats = model.causal_lm_loss(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        tokens = int(stats["target_tokens"].item())
                    else:
                        decoder_prompt = None
                        decoder_mask = None
                        if decoder_mode == "memory-only":
                            decoder_prompt = torch.full(
                                (batch["prompt_ids"].size(0), 1),
                                bos_token_id,
                                dtype=torch.long,
                                device=device,
                            )
                            decoder_mask = torch.ones_like(decoder_prompt)
                        loss, stats = model(
                            batch["prompt_ids"],
                            batch["answer_ids_with_bos"],
                            prompt_attention_mask=batch["prompt_attention_mask"],
                            answer_attention_mask=batch["answer_attention_mask"],
                            decoder_prompt_ids=decoder_prompt,
                            decoder_prompt_attention_mask=decoder_mask,
                            reasoning_steps=reasoning_steps,
                        )
                        tokens = int(batch["answer_labels"].ne(-100).sum().item())
                weighted_loss += float(loss.detach()) * tokens
                weighted_accuracy += float(stats["token_accuracy"].detach()) * tokens
                target_tokens += tokens
    finally:
        model.train(was_training)
    if target_tokens == 0:
        raise ValueError("validation produced no target tokens")
    mean = weighted_loss / target_tokens
    return {
        "loss": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "token_accuracy": weighted_accuracy / target_tokens,
        "target_tokens": float(target_tokens),
    }


def train_loop(
    *,
    args: argparse.Namespace,
    stage: str,
    tokenizer: BPETokenizer,
    model: FOGLatentReasoner,
    config: FOGReasonerConfig,
    source: CyclingBatchSource,
    validation_factory: Callable[[], Iterable[dict[str, torch.Tensor]]],
    device: torch.device,
    precision: str,
) -> None:
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(precision == "fp16" and device.type == "cuda")
    )
    global_step = 0
    consumed_tokens = 0
    best_validation = float("inf")
    if args.resume:
        payload = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer_path=args.tokenizer,
            restore_rng=True,
        )
        global_step = int(payload["global_step"])
        consumed_tokens = int(payload["consumed_tokens"])
        best_validation = float(payload.get("extra", {}).get("best_validation", float("inf")))
    model.train()
    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    interval_started = started
    interval_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        aggregate_loss = 0.0
        aggregate_correct = 0.0
        aggregate_tokens = 0
        for _ in range(args.gradient_accumulation):
            batch = move_batch(next(source), device)
            with autocast_context(device, precision):
                if stage == "pretrain":
                    loss, stats = model.causal_lm_loss(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    tokens = int(stats["target_tokens"].item())
                else:
                    decoder_prompt = None
                    decoder_mask = None
                    if args.decoder_mode == "memory-only":
                        decoder_prompt = torch.full(
                            (batch["prompt_ids"].size(0), 1),
                            tokenizer.bos_token_id,
                            dtype=torch.long,
                            device=device,
                        )
                        decoder_mask = torch.ones_like(decoder_prompt)
                    loss, stats = model(
                        batch["prompt_ids"],
                        batch["answer_ids_with_bos"],
                        prompt_attention_mask=batch["prompt_attention_mask"],
                        answer_attention_mask=batch["answer_attention_mask"],
                        decoder_prompt_ids=decoder_prompt,
                        decoder_prompt_attention_mask=decoder_mask,
                        reasoning_steps=args.reasoning_steps,
                    )
                    tokens = int(batch["answer_labels"].ne(-100).sum().item())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")
            # Backpropagate a sum over target tokens.  Gradients are divided by
            # the exact aggregate token count after all microbatches.
            scaler.scale(loss * tokens).backward()
            aggregate_loss += float(loss.detach()) * tokens
            aggregate_correct += float(stats["token_accuracy"].detach()) * tokens
            aggregate_tokens += tokens

        scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(aggregate_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {global_step}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        consumed_tokens += aggregate_tokens
        interval_tokens += aggregate_tokens

        if global_step == 1 or global_step % args.log_every == 0:
            now = time.perf_counter()
            record = {
                "stage": stage,
                "step": global_step,
                "loss": aggregate_loss / aggregate_tokens,
                "perplexity": math.exp(min(aggregate_loss / aggregate_tokens, 20.0)),
                "token_accuracy": aggregate_correct / aggregate_tokens,
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "tokens": consumed_tokens,
                "tokens_per_second": interval_tokens / max(now - interval_started, 1e-9),
            }
            print(json.dumps(record), flush=True)
            interval_started = now
            interval_tokens = 0

        validation: dict[str, float] | None = None
        if global_step == args.max_steps or (
            args.eval_every and global_step % args.eval_every == 0
        ):
            validation = finite_eval(
                model,
                validation_factory(),
                device=device,
                precision=precision,
                stage=stage,
                decoder_mode=("full" if stage == "pretrain" else args.decoder_mode),
                bos_token_id=tokenizer.bos_token_id,
                max_batches=args.eval_batches,
                reasoning_steps=(None if stage == "pretrain" else args.reasoning_steps),
            )
            print(json.dumps({"stage": stage, "step": global_step, "validation": validation}))

        is_best = validation is not None and validation["loss"] < best_validation
        if is_best:
            best_validation = validation["loss"]
        should_save = global_step % args.save_every == 0 or global_step == args.max_steps
        if should_save:
            extra = {
                "stage": stage,
                "best_validation": best_validation,
                "data_state": source.state_dict(),
                "precision": precision,
                "elapsed_seconds": time.perf_counter() - started,
            }
            save_training_checkpoint(
                output_dir / "last.pt",
                model=model,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                consumed_tokens=consumed_tokens,
                tokenizer_path=args.tokenizer,
                training_args=serializable_args(args),
                extra=extra,
            )
        if is_best:
            save_training_checkpoint(
                output_dir / "best.pt",
                model=model,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                consumed_tokens=consumed_tokens,
                tokenizer_path=args.tokenizer,
                training_args=serializable_args(args),
                extra={
                    "stage": stage,
                    "best_validation": best_validation,
                    "data_state": source.state_dict(),
                    "validation": validation,
                },
            )


def make_text_batches(
    args: argparse.Namespace, tokenizer: BPETokenizer
) -> tuple[CyclingBatchSource, Callable[[], Iterable[dict[str, torch.Tensor]]]]:
    collator = TextBlockCollator(tokenizer, block_size=args.sequence_length)
    if args.local_data:
        train_factory = shuffled_local_factory(
            args.local_data, required_fields=(args.text_field,), seed=args.seed
        )
        validation_factory_records = (
            shuffled_local_factory(
                args.local_eval_data,
                required_fields=(args.text_field,),
                seed=args.seed + 100_000,
            )
            if args.local_eval_data
            else train_factory
        )
    else:
        train_factory = hf_factory(
            dataset_id=args.dataset_id,
            config=args.dataset_config,
            split=args.train_split,
            revision=args.revision,
            streaming=args.streaming,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        validation_factory_records = hf_factory(
            dataset_id=args.dataset_id,
            config=args.dataset_config,
            split=args.eval_split,
            revision=args.revision,
            streaming=args.streaming,
            seed=args.seed + 100_000,
            shuffle_buffer=args.shuffle_buffer,
        )
    state = {"epoch": 0, "batch_in_epoch": 0}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        state.update(payload.get("extra", {}).get("data_state", {}))
    source = CyclingBatchSource(
        train_factory,
        collator,
        args.batch_size,
        epoch=state["epoch"],
        batch_in_epoch=state["batch_in_epoch"],
    )

    def validation_batches():
        for records in chunked(validation_factory_records(0), args.batch_size):
            try:
                yield collator(records)
            except ValueError as exc:
                if "fewer than two usable text tokens" not in str(exc):
                    raise

    return source, validation_batches


def split_sft_records(args: argparse.Namespace):
    if args.local_data:
        records: Any = list(
            iter_local_records(
                args.local_data,
                required_fields=(args.prompt_field, args.response_field),
            )
        )
        if args.local_eval_data:
            train = records
            validation = list(
                iter_local_records(
                    args.local_eval_data,
                    required_fields=(args.prompt_field, args.response_field),
                )
            )
            if not validation:
                raise ValueError("local SFT validation dataset is empty")
        else:
            if len(records) <= args.validation_size:
                raise ValueError("local SFT dataset must be larger than validation_size")
            order = list(range(len(records)))
            random.Random(args.seed).shuffle(order)
            validation = [records[index] for index in order[: args.validation_size]]
            train = [records[index] for index in order[args.validation_size :]]
    else:
        records = load_hf_dataset(
            args.dataset_id,
            config=args.dataset_config,
            split=args.train_split,
            revision=args.revision,
            streaming=False,
        )
        split = records.train_test_split(
            test_size=args.validation_size, seed=args.seed, shuffle=True
        )
        train, validation = split["train"], split["test"]
    if args.max_train_examples:
        if args.max_train_examples < 1:
            raise ValueError("max_train_examples must be >= 1")
        if hasattr(train, "select"):
            train = train.select(range(min(args.max_train_examples, len(train))))
        else:
            train = train[: args.max_train_examples]
    return train, validation


def make_sft_batches(
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    config: FOGReasonerConfig,
) -> tuple[CyclingBatchSource, Callable[[], Iterable[dict[str, torch.Tensor]]]]:
    worst_decoder_prompt = args.max_prompt_length if args.decoder_mode == "full" else 1
    effective_memory = config.effective_memory_slots()
    worst_total = worst_decoder_prompt + effective_memory + args.max_answer_length - 1
    if worst_total > config.max_seq_len:
        raise ValueError(
            "padded decoder prompt + latent memory + answer exceeds max_seq_len: "
            f"{worst_decoder_prompt} + {effective_memory} + "
            f"{args.max_answer_length - 1} = {worst_total} > {config.max_seq_len}"
        )
    train_records, validation_records = split_sft_records(args)
    collator = PromptResponseCollator(
        tokenizer,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        target_mode=args.target_mode,
        prompt_template=args.prompt_template,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        # Latent memory occupies real sequence positions inside the decoder.
        max_sequence_length=config.max_seq_len - effective_memory,
    )

    def train_factory(epoch: int):
        if hasattr(train_records, "shuffle"):
            return iter(train_records.shuffle(seed=args.seed + epoch))
        order = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(order)
        return (train_records[index] for index in order)

    def validation_factory_records():
        return iter(validation_records)

    state = {"epoch": 0, "batch_in_epoch": 0}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        state.update(payload.get("extra", {}).get("data_state", {}))
    source = CyclingBatchSource(
        train_factory,
        collator,
        args.batch_size,
        epoch=state["epoch"],
        batch_in_epoch=state["batch_in_epoch"],
    )

    def validation_batches():
        for records in chunked(validation_factory_records(), args.batch_size):
            yield collator(records)

    return source, validation_batches


def run_pretrain(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    tokenizer = tokenizer_from_args(args)
    model, config = make_model(args, tokenizer, device)
    source, validation = make_text_batches(args, tokenizer)
    print(
        json.dumps(
            {
                "stage": "pretrain",
                "device": str(device),
                "precision": precision,
                "parameters": unique_parameter_count(model),
                "config": asdict(config),
            }
        )
    )
    train_loop(
        args=args,
        stage="pretrain",
        tokenizer=tokenizer,
        model=model,
        config=config,
        source=source,
        validation_factory=validation,
        device=device,
        precision=precision,
    )


def run_sft(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    tokenizer = tokenizer_from_args(args)
    model, config = make_model(args, tokenizer, device)
    source, validation = make_sft_batches(args, tokenizer, config)
    print(
        json.dumps(
            {
                "stage": "sft",
                "device": str(device),
                "precision": precision,
                "decoder_mode": args.decoder_mode,
                "target_mode": args.target_mode,
                "parameters": unique_parameter_count(model),
                "config": asdict(config),
            }
        )
    )
    train_loop(
        args=args,
        stage="sft",
        tokenizer=tokenizer,
        model=model,
        config=config,
        source=source,
        validation_factory=validation,
        device=device,
        precision=precision,
    )


_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")


def normalize_numeric_answer(text: str) -> str:
    matches = _NUMBER.findall(text.replace("$", ""))
    if not matches:
        return text.strip().lower()
    value = matches[-1].replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def evaluate_gsm8k(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    tokenizer = tokenizer_from_args(args)
    config = config_from_checkpoint(args.checkpoint)
    model = FOGLatentReasoner(config).to(device)
    load_training_checkpoint(
        args.checkpoint,
        model=model,
        tokenizer_path=args.tokenizer,
        restore_rng=False,
    )
    records = load_hf_dataset(
        args.dataset_id,
        config=args.dataset_config,
        split=args.test_split,
        revision=args.revision,
        streaming=False,
    )
    worst_decoder_prompt = args.max_prompt_length if args.decoder_mode == "full" else 1
    effective_memory = config.effective_memory_slots()
    worst_total = worst_decoder_prompt + effective_memory + args.max_new_tokens
    if worst_total > config.max_seq_len:
        raise ValueError(
            "padded decoder prompt + latent memory + generation budget exceeds "
            f"max_seq_len: {worst_total} > {config.max_seq_len}"
        )
    if args.max_examples:
        records = records.select(range(min(args.max_examples, len(records))))
    collator = PromptResponseCollator(
        tokenizer,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        target_mode="final",
        prompt_template=args.prompt_template,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        max_sequence_length=config.max_seq_len - effective_memory,
    )
    correct = 0
    total = 0
    examples: list[dict[str, str]] = []
    for rows in chunked(iter(records), args.batch_size):
        batch = move_batch(collator(rows), device)
        decoder_prompt = None
        decoder_prompt_mask = None
        if args.decoder_mode == "memory-only":
            decoder_prompt = torch.full(
                (len(rows), 1), tokenizer.bos_token_id, dtype=torch.long, device=device
            )
            decoder_prompt_mask = torch.ones_like(decoder_prompt)
        with autocast_context(device, precision):
            generated, _ = model.generate(
                batch["prompt_ids"],
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                prompt_attention_mask=batch["prompt_attention_mask"],
                decoder_prompt_ids=decoder_prompt,
                decoder_prompt_attention_mask=decoder_prompt_mask,
            )
        for row, token_ids in zip(rows, generated.tolist(), strict=True):
            prediction = tokenizer.decode(token_ids)
            target = extract_gsm8k_final(row[args.response_field])
            is_correct = normalize_numeric_answer(prediction) == normalize_numeric_answer(target)
            correct += int(is_correct)
            total += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "prediction": prediction,
                        "target": target,
                        "correct": str(is_correct),
                    }
                )
    result = {
        "dataset": args.dataset_id,
        "split": args.test_split,
        "examples": total,
        "exact_match": correct / max(total, 1),
        "samples": examples,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def add_tokenizer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--allow-nonstandard-vocab", action="store_true")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    add_tokenizer_argument(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--architecture",
        choices=("auto", "legacy_v1", "query_bound_v2", "register_machine_v3"),
        default="auto",
        help="new model architecture; auto follows a checkpoint or selects query_bound_v2",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=250)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    token = commands.add_parser("tokenizer", help="train the 8192-token byte BPE")
    token.add_argument("--local-data", nargs="+")
    token.add_argument(
        "--offline-byte-fallback",
        action="store_true",
        help="create a lossless 8192-ID byte tokenizer without downloading data",
    )
    token.add_argument("--dataset-id", default=TINYSTORIES_ID)
    token.add_argument("--dataset-config", default=TINYSTORIES_CONFIG)
    token.add_argument("--revision", default=TINYSTORIES_REVISION)
    token.add_argument("--train-split", default="train")
    token.add_argument("--text-field", default="text")
    token.add_argument("--shuffle-buffer", type=int, default=50_000)
    token.add_argument("--max-samples", type=int, default=50_000)
    token.add_argument("--vocab-size", type=int, default=FOG_10M_VOCAB_SIZE)
    token.add_argument("--min-frequency", type=int, default=2)
    token.add_argument("--seed", type=int, default=42)
    token.add_argument("--output", default="tokenizer/tokenizer.json")
    token.set_defaults(func=train_tokenizer)

    initial = commands.add_parser("init-model", help="write an initialized 10M checkpoint")
    add_tokenizer_argument(initial)
    initial.add_argument("--seed", type=int, default=42)
    initial.add_argument("--max-seq-len", type=int, default=512)
    initial.add_argument("--reasoning-steps", type=int, default=4)
    initial.add_argument(
        "--architecture",
        choices=("legacy_v1", "query_bound_v2", "register_machine_v3"),
        default="query_bound_v2",
    )
    initial.add_argument("--dropout", type=float, default=0.1)
    initial.add_argument("--output", default="checkpoints/fog_10m_init.pt")
    initial.set_defaults(func=init_model)

    pretrain = commands.add_parser("pretrain", help="causal-LM pretraining on TinyStories")
    add_training_arguments(pretrain)
    pretrain.add_argument("--local-data", nargs="+")
    pretrain.add_argument(
        "--local-eval-data",
        nargs="+",
        help="separate local TXT/JSONL validation files",
    )
    pretrain.add_argument("--dataset-id", default=TINYSTORIES_ID)
    pretrain.add_argument("--dataset-config", default=TINYSTORIES_CONFIG)
    pretrain.add_argument("--revision", default=TINYSTORIES_REVISION)
    pretrain.add_argument("--train-split", default="train")
    pretrain.add_argument("--eval-split", default="validation")
    pretrain.add_argument("--text-field", default="text")
    pretrain.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    pretrain.add_argument("--shuffle-buffer", type=int, default=50_000)
    pretrain.add_argument("--sequence-length", type=int, default=256)
    pretrain.set_defaults(func=run_pretrain)

    sft = commands.add_parser("sft", help="answer-only latent SFT on GSM8K")
    add_training_arguments(sft)
    sft.add_argument("--local-data", nargs="+")
    sft.add_argument(
        "--local-eval-data",
        nargs="+",
        help="separate local JSONL validation files (never mixed into train)",
    )
    sft.add_argument("--dataset-id", default=GSM8K_ID)
    sft.add_argument("--dataset-config", default=GSM8K_CONFIG)
    sft.add_argument("--revision", default=GSM8K_REVISION)
    sft.add_argument("--train-split", default="train")
    sft.add_argument("--prompt-field", default="question")
    sft.add_argument("--response-field", default="answer")
    sft.add_argument("--prompt-template", default="Question: {prompt}\nAnswer:")
    sft.add_argument("--target-mode", choices=("final", "full"), default="final")
    sft.add_argument("--decoder-mode", choices=("memory-only", "full"), default="memory-only")
    sft.add_argument("--validation-size", type=int, default=512)
    sft.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="cap the post-split train set; useful for an overfit gate",
    )
    sft.add_argument("--max-prompt-length", type=int, default=320)
    sft.add_argument("--max-answer-length", type=int, default=176)
    sft.add_argument("--reasoning-steps", type=int, default=4)
    sft.set_defaults(func=run_sft)

    evaluate = commands.add_parser("evaluate-gsm8k", help="official GSM8K test exact match")
    add_tokenizer_argument(evaluate)
    add_runtime_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset-id", default=GSM8K_ID)
    evaluate.add_argument("--dataset-config", default=GSM8K_CONFIG)
    evaluate.add_argument("--revision", default=GSM8K_REVISION)
    evaluate.add_argument("--test-split", default="test")
    evaluate.add_argument("--prompt-field", default="question")
    evaluate.add_argument("--response-field", default="answer")
    evaluate.add_argument("--prompt-template", default="Question: {prompt}\nAnswer:")
    evaluate.add_argument("--decoder-mode", choices=("memory-only", "full"), default="memory-only")
    evaluate.add_argument("--max-prompt-length", type=int, default=320)
    evaluate.add_argument("--max-answer-length", type=int, default=176)
    evaluate.add_argument("--max-new-tokens", type=int, default=48)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--max-examples", type=int, default=0)
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=evaluate_gsm8k)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "sequence_length") and args.sequence_length > args.max_seq_len:
        raise ValueError("sequence_length cannot exceed max_seq_len")
    if hasattr(args, "gradient_accumulation") and args.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    for name in ("batch_size", "log_every", "save_every", "eval_batches"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
