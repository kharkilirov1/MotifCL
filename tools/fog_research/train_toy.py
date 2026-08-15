"""Train and evaluate FOG on dynamic pointer chasing.

Every example contains a newly sampled 8-state permutation.  The target is the
state reached after applying that permutation L times, so memorising a global
lookup table cannot solve the task.  The recurrent model uses one latent
iteration per hop; the one-shot control has identical weights and always uses
one iteration.  Training uses L=1..4 and length OOD evaluation uses L=5..8.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Iterable, NamedTuple

import torch
from torch.nn import functional as F

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner


PAD = 0
ANSWER_BOS = 1
DB = 2
QUERY = 3
STEPS = 4
END = 5
STATE_BASE = 10
N_STATES = 8
MAX_PROGRAM = 8
STEP_BASE = STATE_BASE + N_STATES
VOCAB_SIZE = STEP_BASE + MAX_PROGRAM


class Example(NamedTuple):
    prompt: list[int]
    target: int
    length: int
    intermediate: list[int]


def make_example(seed: int, length: int, mode: str) -> Example:
    if not 1 <= length <= MAX_PROGRAM:
        raise ValueError(f"length must be in [1, {MAX_PROGRAM}]")
    rng = random.Random(seed)
    if mode in {"train", "id", "length"}:
        transition = list(range(N_STATES))
        rng.shuffle(transition)
    elif mode in {"function", "function_length"}:
        # Structural OOD: a general (possibly many-to-one) finite function.
        transition = [rng.randrange(N_STATES) for _ in range(N_STATES)]
    else:
        raise ValueError(f"unknown example mode: {mode}")
    start = rng.randrange(N_STATES)
    state = start
    intermediate = []
    for _ in range(length):
        state = transition[state]
        intermediate.append(state)

    prompt = [DB]
    # Source state is represented by the fixed table position; token value is
    # its destination. This makes one-hop lookup learnable while composition
    # still requires applying the freshly sampled operator repeatedly.
    prompt.extend(STATE_BASE + destination for destination in transition)
    prompt.extend([QUERY, STATE_BASE + start, STEPS, STEP_BASE + length - 1, END])
    if len(prompt) != 14:
        raise AssertionError(f"unexpected prompt length {len(prompt)}")
    return Example(prompt, STATE_BASE + state, length, intermediate)


def make_examples(
    *, base_seed: int, lengths: Iterable[int], count: int, mode: str
) -> list[Example]:
    lengths = tuple(lengths)
    return [
        make_example(base_seed + index * 1_000_003, lengths[index % len(lengths)], mode)
        for index in range(count)
    ]


def make_train_batch(
    *, step: int, length: int, batch_size: int, base_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [
        make_example(
            base_seed + step * 100_003 + index * 1_009,
            length,
            "train",
        )
        for index in range(batch_size)
    ]
    prompt = torch.tensor([example.prompt for example in examples], dtype=torch.long)
    answer = torch.tensor(
        [[ANSWER_BOS, example.target] for example in examples], dtype=torch.long
    )
    return prompt, answer


def curriculum_length(step: int) -> int:
    if step < 100:
        return 1
    if step < 250:
        return 1 + (step - 100) % 2
    return 1 + (step - 250) % 4


def model_config() -> FOGReasonerConfig:
    return FOGReasonerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=48,
        n_heads=4,
        n_layers=1,
        d_ff=96,
        max_seq_len=128,
        dropout=0.0,
        latent_slots=4,
        reasoning_steps=4,
        compare_rank=12,
        planner_ff=96,
        memory_slots=4,
        n_reasoning_modes=3,
        diversity_weight=1e-3,
        route_entropy_weight=0.0,
    )


def reasoning_depth(variant: str, length: int) -> int:
    if variant == "recurrent":
        return length
    if variant == "one_shot":
        return 1
    if variant == "no_latent":
        return 0
    raise ValueError(f"unknown variant: {variant}")


@torch.inference_mode()
def evaluate(
    model: FOGLatentReasoner,
    examples: list[Example],
    *,
    variant: str,
    batch_size: int = 128,
    depth_override: int | None = None,
    memory_intervention: str = "normal",
) -> dict:
    was_training = model.training
    model.eval()
    total = correct = 0
    total_nll = 0.0
    by_length: dict[int, list[int]] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompt = torch.tensor([example.prompt for example in batch], dtype=torch.long)
        target = torch.tensor([example.target for example in batch], dtype=torch.long)
        length = batch[0].length
        if any(example.length != length for example in batch):
            raise ValueError("evaluation batches must be homogeneous in program length")
        steps = depth_override if depth_override is not None else reasoning_depth(variant, length)
        memory, _ = model.reason(
            prompt, reasoning_steps=steps, return_diagnostics=False
        )
        if memory is not None and memory_intervention == "zero":
            memory = torch.zeros_like(memory)
        elif memory is not None and memory_intervention == "shuffle":
            # Deterministic cyclic permutation avoids an extra RNG dependency.
            memory = memory.roll(1, dims=0)
        elif memory_intervention != "normal":
            raise ValueError(f"unknown intervention: {memory_intervention}")
        decoder = torch.full((len(batch), 1), ANSWER_BOS, dtype=torch.long)
        logits = model.decode(prompt, memory, decoder)[:, 0, :]
        prediction = logits.argmax(dim=-1)
        batch_correct = prediction.eq(target)
        total += len(batch)
        correct += int(batch_correct.sum())
        total_nll += float(F.cross_entropy(logits, target, reduction="sum"))
        counts = by_length.setdefault(length, [0, 0])
        counts[0] += int(batch_correct.sum())
        counts[1] += len(batch)
    model.train(was_training)
    return {
        "accuracy": correct / total,
        "nll": total_nll / total,
        "count": total,
        "by_length": {
            str(length): hits / count for length, (hits, count) in sorted(by_length.items())
        },
        "depth_override": depth_override,
        "memory_intervention": memory_intervention,
    }


def _balanced_evaluate(
    model: FOGLatentReasoner,
    examples: list[Example],
    *,
    variant: str,
    batch_size: int,
    depth_override: int | None = None,
    memory_intervention: str = "normal",
) -> dict:
    # Grouping preserves homogeneous depth within every batch.
    parts = []
    for length in sorted({example.length for example in examples}):
        subset = [example for example in examples if example.length == length]
        parts.append(
            evaluate(
                model,
                subset,
                variant=variant,
                batch_size=batch_size,
                depth_override=depth_override,
                memory_intervention=memory_intervention,
            )
        )
    count = sum(part["count"] for part in parts)
    return {
        "accuracy": sum(part["accuracy"] * part["count"] for part in parts) / count,
        "nll": sum(part["nll"] * part["count"] for part in parts) / count,
        "count": count,
        "by_length": {
            length: accuracy
            for part in parts
            for length, accuracy in part["by_length"].items()
        },
        "depth_override": depth_override,
        "memory_intervention": memory_intervention,
    }


def _example_digest(example: Example) -> str:
    payload = bytes(example.prompt + [example.target])
    return hashlib.sha256(payload).hexdigest()


def build_eval_sets(count: int) -> dict[str, list[Example]]:
    count = max(8, count - count % 8)
    sets = {
        "val_id": make_examples(
            base_seed=2_201, lengths=(1, 2, 3, 4), count=count, mode="id"
        ),
        "test_id": make_examples(
            base_seed=3_301, lengths=(1, 2, 3, 4), count=count, mode="id"
        ),
        "test_length": make_examples(
            base_seed=4_401, lengths=(5, 6, 7, 8), count=count, mode="length"
        ),
        "test_function": make_examples(
            base_seed=5_501, lengths=(1, 2, 3, 4), count=count, mode="function"
        ),
        "test_function_length": make_examples(
            base_seed=6_601,
            lengths=(5, 6, 7, 8),
            count=count,
            mode="function_length",
        ),
    }
    digests: dict[str, set[str]] = {
        name: {_example_digest(example) for example in examples}
        for name, examples in sets.items()
    }
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = digests[left] & digests[right]
            if overlap:
                raise AssertionError(f"{left}/{right} share {len(overlap)} full examples")
    return sets


def train_variant(
    *,
    variant: str,
    steps: int,
    batch_size: int,
    model_seed: int,
    train_seed: int,
    eval_sets: dict[str, list[Example]],
    output_dir: Path,
    log_every: int,
) -> tuple[FOGLatentReasoner, dict]:
    torch.manual_seed(model_seed)
    cfg = model_config()
    model = FOGLatentReasoner(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-2,
    )

    def lr_factor(index: int) -> float:
        if index < 50:
            return max((index + 1) / 50, 1e-3)
        progress = min((index - 50) / max(steps - 50, 1), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        length = curriculum_length(step)
        prompt, answer = make_train_batch(
            step=step,
            length=length,
            batch_size=batch_size,
            base_seed=train_seed,
        )
        depth = reasoning_depth(variant, length)
        optimizer.zero_grad(set_to_none=True)
        loss, aux = model(
            prompt,
            answer,
            reasoning_steps=depth,
            return_diagnostics=False,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            val_subset = eval_sets["val_id"][: min(256, len(eval_sets["val_id"]))]
            val = _balanced_evaluate(
                model,
                val_subset,
                variant=variant,
                batch_size=128,
            )
            row = {
                "step": step + 1,
                "length": length,
                "loss": float(loss.detach()),
                "ce": float(aux["ce_loss"].detach()),
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "val_accuracy": val["accuracy"],
            }
            trace.append(row)
            print(
                f"[{variant:9s}] step={step + 1:4d}/{steps} "
                f"L={length} ce={row['ce']:.4f} "
                f"val={100 * row['val_accuracy']:.1f}% lr={row['lr']:.2e}",
                flush=True,
            )

    train_seconds = time.perf_counter() - started
    metrics = {
        "variant": variant,
        "model_seed": model_seed,
        "train_seed": train_seed,
        "steps": steps,
        "batch_size": batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": train_seconds,
        "trace": trace,
        "splits": {},
    }
    for name, examples in eval_sets.items():
        metrics["splits"][name] = _balanced_evaluate(
            model, examples, variant=variant, batch_size=128
        )

    if variant == "recurrent":
        metrics["test_length_depth_1"] = _balanced_evaluate(
            model,
            eval_sets["test_length"],
            variant=variant,
            batch_size=128,
            depth_override=1,
        )
        metrics["test_length_depth_4"] = _balanced_evaluate(
            model,
            eval_sets["test_length"],
            variant=variant,
            batch_size=128,
            depth_override=4,
        )
        for split in ("test_id", "test_length"):
            for intervention in ("zero", "shuffle"):
                metrics[f"{split}_memory_{intervention}"] = _balanced_evaluate(
                    model,
                    eval_sets[split],
                    variant=variant,
                    batch_size=128,
                    memory_intervention=intervention,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{variant}.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "task": "dynamic_pointer_chasing_v1",
            "tokens": {
                "answer_bos": ANSWER_BOS,
                "state_base": STATE_BASE,
                "step_base": STEP_BASE,
                "n_states": N_STATES,
            },
            "training": {
                "variant": variant,
                "steps": steps,
                "batch_size": batch_size,
                "model_seed": model_seed,
                "train_seed": train_seed,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return model, metrics


def write_report(results: dict, output_dir: Path) -> None:
    lines = [
        "# FOG toy-training report",
        "",
        "Task: repeatedly apply a fresh random 8-state permutation. "
        "Chance accuracy is 12.5%.",
        "",
        "| variant | val-ID | test-ID | length 5-8 | function OOD | function+length | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in results.items():
        split = metrics["splits"]
        lines.append(
            f"| {variant} | {100*split['val_id']['accuracy']:.2f}% | "
            f"{100*split['test_id']['accuracy']:.2f}% | "
            f"{100*split['test_length']['accuracy']:.2f}% | "
            f"{100*split['test_function']['accuracy']:.2f}% | "
            f"{100*split['test_function_length']['accuracy']:.2f}% | "
            f"{metrics['train_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The recurrent variant uses R=program length; the one-shot control has "
            "the same architecture and parameter count but always uses R=1.",
            "",
            "This is a CPU-scale functional experiment, not evidence of superiority "
            "to language-model baselines. See metrics.json for per-length results and "
            "memory interventions.",
        ]
    )
    (output_dir / "TRAINING_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=1_020)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=1_101)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("recurrent", "one_shot", "no_latent"),
        default=("recurrent", "one_shot"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/toy_seed0"))
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0 or args.eval_examples <= 0:
        raise ValueError("steps, batch-size, and eval-examples must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    eval_sets = build_eval_sets(args.eval_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in args.variants:
        _, results[variant] = train_variant(
            variant=variant,
            steps=args.steps,
            batch_size=args.batch_size,
            model_seed=args.model_seed,
            train_seed=args.train_seed,
            eval_sets=eval_sets,
            output_dir=args.output_dir,
            log_every=args.log_every,
        )

    payload = {
        "experiment": "dynamic_pointer_chasing_v1",
        "chance_accuracy": 1 / N_STATES,
        "model_config": asdict(model_config()),
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_report(results, args.output_dir)
    print(json.dumps({
        variant: {
            split: round(100 * result["splits"][split]["accuracy"], 2)
            for split in (
                "test_id",
                "test_length",
                "test_function",
                "test_function_length",
            )
        }
        for variant, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
