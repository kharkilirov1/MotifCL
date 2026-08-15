"""Mechanistic end-to-end training for the FOG latent recurrence.

The prompt contains an operator and a start state, but deliberately omits the
requested hop count.  The hop count is represented only by the number of latent
iterations R.  Consequently, examples with the same lexical prompt can have
different targets and a decoder that ignores latent memory cannot solve them.

This is a sanity experiment for recurrence and memory use, not a language-model
benchmark.  All intermediate iterations stay latent and only the final state is
decoded/supervised.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner


PAD = 0
ANSWER_BOS = 1
TASK = 2
END = 3
STATE_BASE = 4
N_STATES = 8
OP_BASE = STATE_BASE + N_STATES
SHIFTS = (1, 2, 3)
VOCAB_SIZE = OP_BASE + len(SHIFTS)


def transition(operator: int, state: int) -> int:
    return (state + SHIFTS[operator]) % N_STATES


def target_state(operator: int, start: int, length: int) -> int:
    state = start
    for _ in range(length):
        state = transition(operator, state)
    return state


def encode_prompt(operator: int, start: int) -> list[int]:
    # No length token: only latent recurrence depth communicates hop count.
    return [TASK, OP_BASE + operator, STATE_BASE + start, END]


def make_batch(
    *, seed: int, step: int, length: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed + step * 1_000_003 + length * 10_007)
    prompts = []
    answers = []
    for _ in range(batch_size):
        operator = rng.randrange(len(SHIFTS))
        start = rng.randrange(N_STATES)
        prompts.append(encode_prompt(operator, start))
        answers.append(
            [ANSWER_BOS, STATE_BASE + target_state(operator, start, length)]
        )
    return torch.tensor(prompts), torch.tensor(answers)


def exhaustive_examples(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = []
    targets = []
    for operator in range(len(SHIFTS)):
        for start in range(N_STATES):
            prompts.append(encode_prompt(operator, start))
            targets.append(STATE_BASE + target_state(operator, start, length))
    return torch.tensor(prompts), torch.tensor(targets)


def model_config() -> FOGReasonerConfig:
    return FOGReasonerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=48,
        n_heads=4,
        n_layers=1,
        d_ff=96,
        max_seq_len=32,
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


def active_lengths(step: int) -> tuple[int, ...]:
    if step < 300:
        return (1,)
    if step < 800:
        return (1, 2)
    return (1, 2, 3, 4)


def depth_for(variant: str, length: int) -> int:
    if variant == "recurrent":
        return length
    if variant == "one_shot":
        return 1
    if variant == "no_latent":
        return 0
    raise ValueError(f"unknown variant: {variant}")


@torch.inference_mode()
def evaluate_lengths(
    model: FOGLatentReasoner,
    lengths: tuple[int, ...],
    *,
    variant: str,
    memory_intervention: str = "normal",
    depth_override: int | None = None,
    decoder_bottleneck: bool = True,
) -> dict:
    was_training = model.training
    model.eval()
    by_length = {}
    total_correct = total_count = 0
    total_nll = 0.0
    for length in lengths:
        prompt, target = exhaustive_examples(length)
        depth = depth_override if depth_override is not None else depth_for(variant, length)
        memory, _ = model.reason(
            prompt, reasoning_steps=depth, return_diagnostics=False
        )
        if memory is not None and memory_intervention == "zero":
            memory = torch.zeros_like(memory)
        elif memory is not None and memory_intervention == "shuffle":
            # Roll across (operator, start) examples while retaining shape/kind.
            memory = memory.roll(1, dims=0)
        elif memory_intervention != "normal":
            raise ValueError(f"unknown memory intervention: {memory_intervention}")
        lexical_prompt = (
            torch.full((prompt.size(0), 1), TASK, dtype=torch.long)
            if decoder_bottleneck
            else prompt
        )
        decoder = torch.full((prompt.size(0), 1), ANSWER_BOS, dtype=torch.long)
        logits = model.decode(lexical_prompt, memory, decoder)[:, 0]
        correct = int(logits.argmax(dim=-1).eq(target).sum())
        count = target.numel()
        nll = float(F.cross_entropy(logits, target, reduction="sum"))
        by_length[str(length)] = {
            "accuracy": correct / count,
            "nll": nll / count,
            "correct": correct,
            "count": count,
        }
        total_correct += correct
        total_count += count
        total_nll += nll
    model.train(was_training)
    return {
        "accuracy": total_correct / total_count,
        "nll": total_nll / total_count,
        "correct": total_correct,
        "count": total_count,
        "by_length": by_length,
        "memory_intervention": memory_intervention,
        "depth_override": depth_override,
        "decoder_bottleneck": decoder_bottleneck,
    }


def train_variant(
    *,
    variant: str,
    steps: int,
    batch_size: int,
    model_seed: int,
    data_seed: int,
    output_dir: Path,
    log_every: int,
    decoder_bottleneck: bool,
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
    trace = []
    best_accuracy = -1.0
    best_step = 0
    best_state = None
    started = time.perf_counter()
    for step in range(steps):
        lengths = active_lengths(step)
        length = lengths[step % len(lengths)]
        prompt, answer = make_batch(
            seed=data_seed,
            step=step,
            length=length,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, aux = model(
            prompt,
            answer,
            decoder_prompt_ids=(
                torch.full((prompt.size(0), 1), TASK, dtype=torch.long)
                if decoder_bottleneck
                else None
            ),
            reasoning_steps=depth_for(variant, length),
            return_diagnostics=False,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            train_eval = evaluate_lengths(
                model,
                (1, 2, 3, 4),
                variant=variant,
                decoder_bottleneck=decoder_bottleneck,
            )
            row = {
                "step": step + 1,
                "sampled_length": length,
                "ce": aux["ce_loss"].detach().item(),
                "loss": loss.detach().item(),
                "grad_norm": float(grad_norm),
                "train_depth_accuracy": train_eval["accuracy"],
            }
            trace.append(row)
            if train_eval["accuracy"] > best_accuracy:
                best_accuracy = train_eval["accuracy"]
                best_step = step + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            print(
                f"[{variant:9s}] step={step + 1:4d}/{steps} "
                f"L={length} ce={row['ce']:.4f} "
                f"depths1-4={100*row['train_depth_accuracy']:.1f}%",
                flush=True,
            )

    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise AssertionError("training did not produce a checkpoint candidate")
    model.load_state_dict(best_state, strict=True)
    metrics = {
        "variant": variant,
        "steps": steps,
        "batch_size": batch_size,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoder_bottleneck": decoder_bottleneck,
        "train_seconds": train_seconds,
        "selected_step": best_step,
        "selected_depths_1_4_accuracy": best_accuracy,
        "trace": trace,
        "depths_1_4": evaluate_lengths(
            model,
            (1, 2, 3, 4),
            variant=variant,
            decoder_bottleneck=decoder_bottleneck,
        ),
        "depths_5_8": evaluate_lengths(
            model,
            (5, 6, 7, 8),
            variant=variant,
            decoder_bottleneck=decoder_bottleneck,
        ),
    }
    if variant == "recurrent":
        for split, lengths in {
            "depths_1_4": (1, 2, 3, 4),
            "depths_5_8": (5, 6, 7, 8),
        }.items():
            for intervention in ("zero", "shuffle"):
                metrics[f"{split}_memory_{intervention}"] = evaluate_lengths(
                    model,
                    lengths,
                    variant=variant,
                    memory_intervention=intervention,
                    decoder_bottleneck=decoder_bottleneck,
                )
        metrics["depths_5_8_at_R1"] = evaluate_lengths(
            model,
            (5, 6, 7, 8),
            variant=variant,
            depth_override=1,
            decoder_bottleneck=decoder_bottleneck,
        )
        metrics["depths_5_8_at_R4"] = evaluate_lengths(
            model,
            (5, 6, 7, 8),
            variant=variant,
            depth_override=4,
            decoder_bottleneck=decoder_bottleneck,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "task": "finite_state_latent_iteration_v1",
            "model_config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "tokens": {
                "answer_bos": ANSWER_BOS,
                "state_base": STATE_BASE,
                "operator_base": OP_BASE,
                "shifts": SHIFTS,
            },
            "training": {
                "variant": variant,
                "steps": steps,
                "batch_size": batch_size,
                "model_seed": model_seed,
                "data_seed": data_seed,
                "decoder_bottleneck": decoder_bottleneck,
            },
            "metrics": metrics,
        },
        output_dir / f"{variant}.pt",
    )
    return model, metrics


def verify_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FOGReasonerConfig(**payload["model_config"])
    model = FOGLatentReasoner(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    variant = payload["training"]["variant"]
    actual = evaluate_lengths(
        model,
        (1, 2, 3, 4),
        variant=variant,
        decoder_bottleneck=payload["training"].get("decoder_bottleneck", False),
    )
    expected = payload["metrics"]["depths_1_4"]
    if actual["correct"] != expected["correct"]:
        raise AssertionError(
            f"checkpoint round-trip changed accuracy: {actual['correct']} != "
            f"{expected['correct']}"
        )
    return actual


def write_report(results: dict[str, dict], output_dir: Path) -> None:
    lines = [
        "# FOG finite-state latent-iteration report",
        "",
        "The prompt includes `(operator, start state)` but no hop count. The hop "
        "count is supplied only as latent recurrence depth `R`; intermediate "
        "states are never decoded or supervised. The final decoder sees only a "
        "neutral task marker plus latent memory, not the original prompt. Chance "
        "accuracy is 12.5%.",
        "",
        "| variant | depths 1–4 | unseen depths 5–8 | parameters | train seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, metrics in results.items():
        lines.append(
            f"| {variant} | {100*metrics['depths_1_4']['accuracy']:.2f}% | "
            f"{100*metrics['depths_5_8']['accuracy']:.2f}% | "
            f"{metrics['parameters']:,} | {metrics['train_seconds']:.1f} |"
        )
    if "recurrent" in results:
        m = results["recurrent"]
        ood = m["depths_5_8"]["by_length"]
        lines.extend(
            [
                "",
                "## Memory/depth interventions",
                "",
                f"- Selected checkpoint: optimizer step {m['selected_step']} "
                f"(best exhaustive depths 1–4 score).",
                f"- Normal memory, depths 1–4: {100*m['depths_1_4']['accuracy']:.2f}%",
                f"- Zeroed memory, depths 1–4: "
                f"{100*m['depths_1_4_memory_zero']['accuracy']:.2f}%",
                f"- Shuffled-across-example memory, depths 1–4: "
                f"{100*m['depths_1_4_memory_shuffle']['accuracy']:.2f}%",
                f"- Unseen targets 5–8 evaluated with correct R=L: "
                f"{100*m['depths_5_8']['accuracy']:.2f}%",
                f"- Same unseen targets forced to R=1: "
                f"{100*m['depths_5_8_at_R1']['accuracy']:.2f}%",
                f"- Same unseen targets capped at R=4: "
                f"{100*m['depths_5_8_at_R4']['accuracy']:.2f}%",
                "- Correct-depth OOD accuracy by length: "
                + ", ".join(
                    f"L={length}: {100*values['accuracy']:.2f}%"
                    for length, values in ood.items()
                ),
            ]
        )
    lines.extend(
        [
            "",
            "This is a deliberately small mechanistic sanity check. Success on "
            "depths 1–4 proves trainability and memory use, but does not by itself "
            "establish language reasoning or depth extrapolation.",
        ]
    )
    (output_dir / "TRAINING_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=1_101)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("recurrent", "one_shot", "no_latent"),
        default=("recurrent", "one_shot"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--full-prompt-decoder",
        action="store_true",
        help="disable the strict latent-memory bottleneck ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/finite_state_bottleneck_seed0"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    decoder_bottleneck = not args.full_prompt_decoder
    for variant in args.variants:
        _, results[variant] = train_variant(
            variant=variant,
            steps=args.steps,
            batch_size=args.batch_size,
            model_seed=args.model_seed,
            data_seed=args.data_seed,
            output_dir=args.output_dir,
            log_every=args.log_every,
            decoder_bottleneck=decoder_bottleneck,
        )
        verify_checkpoint(args.output_dir / f"{variant}.pt")

    metrics = {
        "experiment": "finite_state_latent_iteration_v1",
        "chance_accuracy": 1 / N_STATES,
        "one_shot_information_upper_bound_depths_1_4": 0.25,
        "model_config": asdict(model_config()),
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    write_report(results, args.output_dir)
    print(
        json.dumps(
            {
                variant: {
                    "depths_1_4": round(100 * value["depths_1_4"]["accuracy"], 2),
                    "depths_5_8": round(100 * value["depths_5_8"]["accuracy"], 2),
                }
                for variant, value in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
