from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = ROOT / "tokenizer" / "tinystories_3k_bpe.json"
DEFAULT_INIT = ROOT / "checkpoints" / "fog_machine_v3_10m_init.pt"
DEFAULT_TRAIN = ROOT / "data_cache" / "tinystories_train_3000.jsonl"
DEFAULT_EVAL = ROOT / "data_cache" / "tinystories_validation_300.jsonl"


@dataclass(frozen=True)
class Recipe:
    pretrain_steps: int
    sequence_length: int
    batch_size: int
    grad_accum: int
    warmup_steps: int
    pretrain_lr: float
    sft_plan: tuple[tuple[int, int, str], ...]  # R, max_steps, decoder_mode
    sft_lr: float


RECIPES = {
    "smoke": Recipe(
        pretrain_steps=100,
        sequence_length=128,
        batch_size=2,
        grad_accum=1,
        warmup_steps=5,
        pretrain_lr=3e-4,
        sft_plan=((1, 30, "full"), (2, 50, "memory-only")),
        sft_lr=2e-4,
    ),
    "starter": Recipe(
        pretrain_steps=2_000,
        sequence_length=256,
        batch_size=8,
        grad_accum=4,
        warmup_steps=100,
        pretrain_lr=3e-4,
        sft_plan=((1, 300, "full"), (2, 500, "memory-only"), (4, 800, "memory-only")),
        sft_lr=2e-4,
    ),
    "serious": Recipe(
        pretrain_steps=10_000,
        sequence_length=256,
        batch_size=8,
        grad_accum=4,
        warmup_steps=500,
        pretrain_lr=3e-4,
        sft_plan=(
            (1, 1_000, "full"),
            (2, 1_500, "memory-only"),
            (4, 2_500, "memory-only"),
            (8, 3_000, "memory-only"),
        ),
        sft_lr=1.5e-4,
    ),
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def maybe_install(install: bool, need_datasets: bool, dry_run: bool) -> None:
    missing = []
    if not has_module("tokenizers"):
        missing.append("tokenizers")
    if need_datasets and not has_module("datasets"):
        missing.append("datasets")
    if not missing:
        return
    if dry_run:
        print("[dry-run] missing dependencies would be installed/required: " + ", ".join(missing))
        return
    if install:
        run([sys.executable, "-m", "pip", "install", "-e", ".[train]"], dry_run=dry_run)
        return
    names = ", ".join(missing)
    raise SystemExit(
        f"Missing training dependencies: {names}. Run once with --install, "
        "or manually: pip install -e '.[train]'"
    )


def checkpoint_step(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("global_step", 0))
    except Exception:
        return None


def choose_completed_checkpoint(stage_dir: Path) -> Path:
    best = stage_dir / "best.pt"
    last = stage_dir / "last.pt"
    return best if best.exists() else last


def append_data_args(
    cmd: list[str],
    *,
    local_data: list[Path] | None,
    local_eval: list[Path] | None,
    dataset_id: str | None,
    dataset_config: str | None,
    revision: str | None,
) -> None:
    if local_data:
        cmd += ["--local-data", *[str(p) for p in local_data]]
        if local_eval:
            cmd += ["--local-eval-data", *[str(p) for p in local_eval]]
        return
    if dataset_id:
        cmd += ["--dataset-id", dataset_id]
        if dataset_config:
            cmd += ["--dataset-config", dataset_config]
        if revision:
            cmd += ["--revision", revision]
        return
    raise ValueError("no dataset source")


def parse_paths(values: Iterable[str] | None) -> list[Path] | None:
    if not values:
        return None
    return [Path(v).expanduser().resolve() for v in values]


def inspect_jsonl_fields(paths: list[Path] | None) -> tuple[set[str], int]:
    fields: set[str] = set()
    count = 0
    if not paths:
        return fields, count
    for path in paths:
        if path.suffix.lower() not in {".jsonl", ".json"}:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if not fields:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        fields = set(row)
    return fields, count


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "One-command trainer for the model-ready FOG register_machine_v3. "
            "It runs lexical pretraining first, then optional staged latent SFT."
        )
    )
    p.add_argument("--recipe", choices=RECIPES, default="starter")
    p.add_argument("--run-dir", default="runs/v3_easy")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, ...")
    p.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--install", action="store_true", help="install .[train] dependencies if missing")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    p.add_argument("--reset", action="store_true", help="delete this run-dir before starting")

    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))

    p.add_argument("--text-data", nargs="+", help="local TXT/JSONL files for lexical pretraining")
    p.add_argument("--text-eval-data", nargs="+", help="separate local lexical validation files")
    p.add_argument("--dataset-id", help="Hugging Face dataset id instead of local text files")
    p.add_argument("--dataset-config")
    p.add_argument("--revision")

    p.add_argument("--skip-pretrain", action="store_true")
    p.add_argument("--pretrain-steps", type=int)
    p.add_argument("--sequence-length", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gradient-accumulation", type=int)

    p.add_argument("--sft-data", nargs="+", help="prompt/response JSONL files; omitted = stop after lexical pretrain")
    p.add_argument("--sft-eval-data", nargs="+", help="separate prompt/response validation JSONL")
    p.add_argument("--prompt-field", default="prompt")
    p.add_argument("--response-field", default="response")
    p.add_argument("--sft-validation-size", type=int, default=128)
    p.add_argument(
        "--sft-depths",
        help="override recipe, e.g. 1:300:full,2:500:memory-only,4:800:memory-only",
    )
    args = p.parse_args()

    recipe = RECIPES[args.recipe]
    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    if args.reset and run_dir.exists() and not args.dry_run:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Path(args.tokenizer).expanduser().resolve()
    init_ckpt = Path(args.init_checkpoint).expanduser().resolve()
    if not tokenizer.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer}")
    if not init_ckpt.exists():
        raise SystemExit(f"v3 init checkpoint not found: {init_ckpt}")

    text_data = parse_paths(args.text_data)
    text_eval = parse_paths(args.text_eval_data)
    sft_data = parse_paths(args.sft_data)
    sft_eval = parse_paths(args.sft_eval_data)

    sft_fields, sft_count = inspect_jsonl_fields(sft_data)
    if sft_data and args.prompt_field == "prompt" and args.response_field == "response":
        if not {"prompt", "response"}.issubset(sft_fields) and {"question", "answer"}.issubset(sft_fields):
            args.prompt_field, args.response_field = "question", "answer"
            print("[auto] detected GSM8K-style fields: question / answer", flush=True)
    if sft_data and not sft_eval:
        if sft_count < 2:
            raise SystemExit("SFT data needs at least 2 JSONL records when no --sft-eval-data is supplied")
        args.sft_validation_size = min(
            args.sft_validation_size,
            max(1, sft_count // 10),
            sft_count - 1,
        )
        print(f"[auto] SFT validation_size={args.sft_validation_size} from {sft_count} local records", flush=True)

    if text_data is None and args.dataset_id is None:
        if not DEFAULT_TRAIN.exists():
            raise SystemExit("No --text-data/--dataset-id and bundled TinyStories data is missing")
        text_data = [DEFAULT_TRAIN]
        text_eval = [DEFAULT_EVAL] if DEFAULT_EVAL.exists() else None
        if args.recipe != "smoke":
            print(
                "WARNING: using bundled 3k TinyStories. This is enough for pipeline testing, "
                "not a serious language pretrain. Pass --text-data or --dataset-id for a real run.",
                flush=True,
            )

    maybe_install(args.install, need_datasets=args.dataset_id is not None, dry_run=args.dry_run)

    # Conservative CPU defaults. Explicit CLI values always win.
    auto_cpu = args.device == "auto" and not torch.cuda.is_available()
    batch_size = args.batch_size or (1 if auto_cpu else recipe.batch_size)
    grad_accum = args.gradient_accumulation or (1 if auto_cpu else recipe.grad_accum)
    sequence_length = args.sequence_length or (128 if auto_cpu else recipe.sequence_length)
    pretrain_steps = args.pretrain_steps or recipe.pretrain_steps

    manifest: dict[str, object] = {
        "recipe": args.recipe,
        "device": args.device,
        "precision": args.precision,
        "seed": args.seed,
        "tokenizer": str(tokenizer),
        "init_checkpoint": str(init_ckpt),
        "lexical_source": [str(p) for p in text_data] if text_data else args.dataset_id,
        "sft_source": [str(p) for p in sft_data] if sft_data else None,
        "stages": [],
    }

    current = init_ckpt
    if not args.skip_pretrain:
        stage_dir = run_dir / "01_pretrain"
        last = stage_dir / "last.pt"
        step = checkpoint_step(last)
        if step is not None and step >= pretrain_steps:
            print(f"[skip] lexical pretrain already completed at step {step}")
        else:
            cmd = [
                sys.executable,
                "train_real.py",
                "pretrain",
                "--architecture",
                "register_machine_v3",
                "--tokenizer",
                str(tokenizer),
                "--checkpoint-dir",
                str(stage_dir),
                "--device",
                args.device,
                "--precision",
                args.precision,
                "--seed",
                str(args.seed),
                "--sequence-length",
                str(sequence_length),
                "--batch-size",
                str(batch_size),
                "--gradient-accumulation",
                str(grad_accum),
                "--max-steps",
                str(pretrain_steps),
                "--warmup-steps",
                str(min(recipe.warmup_steps, max(pretrain_steps // 10, 1))),
                "--lr",
                str(recipe.pretrain_lr),
                "--weight-decay",
                "0.01",
                "--eval-every",
                str(max(pretrain_steps // 10, 1)),
                "--eval-batches",
                "10",
                "--save-every",
                str(max(pretrain_steps // 10, 1)),
                "--log-every",
                str(max(pretrain_steps // 50, 1)),
            ]
            if last.exists():
                cmd += ["--resume", str(last)]
            else:
                cmd += ["--init-checkpoint", str(current)]
            append_data_args(
                cmd,
                local_data=text_data,
                local_eval=text_eval,
                dataset_id=args.dataset_id,
                dataset_config=args.dataset_config,
                revision=args.revision,
            )
            run(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            current = choose_completed_checkpoint(stage_dir)
        else:
            current = stage_dir / "best.pt"
        manifest["stages"].append({"name": "pretrain", "target_steps": pretrain_steps, "checkpoint": str(current)})

    if sft_data:
        if args.sft_depths:
            plan = []
            for item in args.sft_depths.split(","):
                depth_s, steps_s, mode = item.split(":", 2)
                if mode not in {"full", "memory-only"}:
                    raise SystemExit(f"bad decoder mode in --sft-depths: {mode}")
                plan.append((int(depth_s), int(steps_s), mode))
            sft_plan = tuple(plan)
        else:
            sft_plan = recipe.sft_plan

        for index, (depth, steps, decoder_mode) in enumerate(sft_plan, start=2):
            stage_dir = run_dir / f"{index:02d}_sft_r{depth}_{decoder_mode.replace('-', '_')}"
            last = stage_dir / "last.pt"
            completed = checkpoint_step(last)
            if completed is not None and completed >= steps:
                print(f"[skip] SFT R={depth} already completed at step {completed}")
            else:
                cmd = [
                    sys.executable,
                    "train_real.py",
                    "sft",
                    "--architecture",
                    "register_machine_v3",
                    "--tokenizer",
                    str(tokenizer),
                    "--checkpoint-dir",
                    str(stage_dir),
                    "--device",
                    args.device,
                    "--precision",
                    args.precision,
                    "--seed",
                    str(args.seed + depth),
                    "--reasoning-steps",
                    str(depth),
                    "--target-mode",
                    "full",
                    "--decoder-mode",
                    decoder_mode,
                    "--prompt-field",
                    args.prompt_field,
                    "--response-field",
                    args.response_field,
                    "--validation-size",
                    str(args.sft_validation_size),
                    "--max-prompt-length",
                    "192",
                    "--max-answer-length",
                    "96",
                    "--batch-size",
                    str(max(1, batch_size // 2)),
                    "--gradient-accumulation",
                    str(max(1, grad_accum)),
                    "--max-steps",
                    str(steps),
                    "--warmup-steps",
                    str(max(1, min(50, steps // 10))),
                    "--lr",
                    str(recipe.sft_lr),
                    "--weight-decay",
                    "0.01",
                    "--eval-every",
                    str(max(steps // 10, 1)),
                    "--eval-batches",
                    "16",
                    "--save-every",
                    str(max(steps // 10, 1)),
                    "--log-every",
                    str(max(steps // 50, 1)),
                    "--local-data",
                    *[str(p) for p in sft_data],
                ]
                if sft_eval:
                    cmd += ["--local-eval-data", *[str(p) for p in sft_eval]]
                if last.exists():
                    cmd += ["--resume", str(last)]
                else:
                    cmd += ["--init-checkpoint", str(current)]
                run(cmd, dry_run=args.dry_run)
            if not args.dry_run:
                current = choose_completed_checkpoint(stage_dir)
            else:
                current = stage_dir / "best.pt"
            manifest["stages"].append(
                {
                    "name": "sft",
                    "reasoning_steps": depth,
                    "decoder_mode": decoder_mode,
                    "target_steps": steps,
                    "checkpoint": str(current),
                }
            )

    manifest["final_checkpoint"] = str(current)
    (run_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n=== FOG v3 training pipeline ready/completed ===")
    print(f"Run directory: {run_dir}")
    print(f"Final checkpoint: {current}")
    if not sft_data:
        print("No --sft-data was supplied, so this run intentionally stops after lexical pretraining.")
        print("Add prompt/response JSONL later and rerun the same command with --sft-data; pretrain will auto-skip/resume.")


if __name__ == "__main__":
    main()
