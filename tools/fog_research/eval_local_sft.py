#!/usr/bin/env python3
"""Evaluate local prompt/answer JSONL with latent-memory interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner
from fog_lmw.checkpoint import load_training_checkpoint
from fog_lmw.data import BPETokenizer, PromptResponseCollator, iter_local_records
from train_real import normalize_numeric_answer


def split_records(records: list[dict], seed: int, validation_size: int):
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    validation = [records[index] for index in order[:validation_size]]
    train = [records[index] for index in order[validation_size:]]
    return train, validation


@torch.inference_mode()
def generate_batch(
    model: FOGLatentReasoner,
    tokenizer: BPETokenizer,
    batch: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    reasoning_steps: int,
    intervention: str,
    decoder_mode: str,
) -> torch.Tensor:
    memory, _ = model.reason(
        batch["prompt_ids"],
        prompt_attention_mask=batch["prompt_attention_mask"],
        reasoning_steps=reasoning_steps,
    )
    if memory is not None and intervention == "zero":
        memory = torch.zeros_like(memory)
    elif memory is not None and intervention == "shuffle":
        memory = memory.roll(1, dims=0)
    elif intervention != "normal":
        raise ValueError(f"unknown intervention {intervention!r}")
    if decoder_mode == "memory-only":
        lexical_prompt = torch.full(
            (batch["prompt_ids"].size(0), 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
        )
        lexical_prompt_mask = None
    elif decoder_mode == "full":
        lexical_prompt = batch["prompt_ids"]
        lexical_prompt_mask = batch["prompt_attention_mask"]
    else:
        raise ValueError(f"unknown decoder_mode {decoder_mode!r}")
    output = torch.full_like(lexical_prompt, tokenizer.bos_token_id)
    finished = torch.zeros(output.size(0), dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits = model.decode(
            lexical_prompt,
            memory,
            output,
            prompt_attention_mask=lexical_prompt_mask,
        )
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        next_token = torch.where(
            finished[:, None],
            torch.full_like(next_token, tokenizer.eos_token_id),
            next_token,
        )
        output = torch.cat([output, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(tokenizer.eos_token_id)
        if bool(finished.all()):
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--reasoning-steps", type=int, default=4)
    parser.add_argument(
        "--decoder-mode", choices=("memory-only", "full"), default="memory-only"
    )
    parser.add_argument(
        "--interventions", nargs="+", choices=("normal", "zero", "shuffle"), default=["normal"]
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = BPETokenizer.load(args.tokenizer)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = FOGReasonerConfig(**payload["model_config"])
    model = FOGLatentReasoner(config)
    load_training_checkpoint(
        args.checkpoint,
        model=model,
        tokenizer_path=args.tokenizer,
        restore_rng=False,
    )
    model.eval()
    records = list(
        iter_local_records(
            args.data, required_fields=("question", "answer")
        )
    )
    train, validation = split_records(records, args.seed, args.validation_size)
    if args.max_train_examples:
        train = train[: args.max_train_examples]
    selected = train if args.split == "train" else validation
    if args.split == "all":
        selected = [*train, *validation]
    if args.max_examples:
        selected = selected[: args.max_examples]
    collator = PromptResponseCollator(
        tokenizer,
        target_mode="final",
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        max_sequence_length=config.max_seq_len - config.memory_slots,
    )
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": payload["global_step"],
        "split": args.split,
        "examples": len(selected),
        "reasoning_steps": args.reasoning_steps,
        "decoder_mode": args.decoder_mode,
        "interventions": {},
    }
    for intervention in args.interventions:
        correct = 0
        samples = []
        started = time.perf_counter()
        for start in range(0, len(selected), args.batch_size):
            rows = selected[start : start + args.batch_size]
            batch = collator(rows)
            generated = generate_batch(
                model,
                tokenizer,
                batch,
                max_new_tokens=args.max_new_tokens,
                reasoning_steps=args.reasoning_steps,
                intervention=intervention,
                decoder_mode=args.decoder_mode,
            )
            for row, token_ids in zip(rows, generated.tolist(), strict=True):
                prediction = tokenizer.decode(token_ids)
                target = row["answer"].rsplit("####", 1)[1].strip()
                matched = normalize_numeric_answer(prediction) == normalize_numeric_answer(target)
                correct += int(matched)
                if len(samples) < 8:
                    samples.append(
                        {
                            "source_row": row.get("_source_row_idx"),
                            "prediction": prediction,
                            "target": target,
                            "correct": matched,
                        }
                    )
        result["interventions"][intervention] = {
            "correct": correct,
            "count": len(selected),
            "exact_match": correct / max(len(selected), 1),
            "seconds": time.perf_counter() - started,
            "samples": samples,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
