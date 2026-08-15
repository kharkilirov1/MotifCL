from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner


def make_model(seed: int = 0, n_states: int = 8) -> FOGLatentReasoner:
    torch.manual_seed(seed)
    cfg = FOGReasonerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        latent_slots=4,
        reasoning_steps=4,
        compare_rank=16,
        planner_ff=64,
        memory_slots=4,
        n_reasoning_modes=2,
        diversity_weight=0.0,
        architecture_version="register_machine_v3",
        binding_mode="query_conditioned",
        readout_mode="direct_latent",
        protected_binding_slots=1,
        binding_offsets=(1,),
        binding_query_update="primary_recurrent",
        machine_operator_count=2,
        machine_operator_rank=16,
        machine_ff=64,
        machine_hard_routing=True,
    )
    model = FOGLatentReasoner(cfg)
    x = torch.arange(n_states, dtype=torch.float32)[:, None]
    k = torch.arange(1, cfg.d_model // 2 + 1, dtype=torch.float32)[None, :]
    angle = 2 * math.pi * x * k / n_states
    code = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1).reshape(n_states, cfg.d_model)
    code = torch.nn.functional.normalize(code, dim=-1) * math.sqrt(cfg.d_model)
    with torch.no_grad():
        model.token.weight[4 : 4 + n_states].copy_(code)
    # Controlled gate: lexical/binding geometry is fixed and already verified;
    # only the recurrent register machine learns the generated-value law.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.machine_cell.parameters():
        parameter.requires_grad_(True)
    return model


def batch(n_states: int, batch_size: int, depth: int, generator: torch.Generator):
    tables = torch.randint(n_states, (batch_size, n_states), generator=generator)
    query = torch.randint(n_states, (batch_size,), generator=generator)
    prompt = torch.zeros(batch_size, 3 * n_states + 1, dtype=torch.long)
    mask = torch.zeros_like(prompt, dtype=torch.bool)
    for state in range(n_states):
        prompt[:, 3 * state] = 4 + state
        prompt[:, 3 * state + 1] = 4 + tables[:, state]
        mask[:, 3 * state] = True
        mask[:, 3 * state + 1] = True
    prompt[:, -1] = 4 + query
    mask[:, -1] = True
    target = query.clone()
    rows = torch.arange(batch_size)
    for _ in range(depth):
        operand = tables[rows, target]
        target = (target + operand) % n_states
    answer = torch.stack([torch.ones_like(target), 4 + target], dim=1)
    return prompt, mask, answer, target


@torch.inference_mode()
def evaluate(model: FOGLatentReasoner, n_states: int, depth: int, seed: int, examples: int = 2048):
    g = torch.Generator().manual_seed(seed)
    correct = 0
    total = 0
    cosine = []
    generated_mass = []
    selected_counts = torch.zeros(model.machine_cell.operator_bank.total_candidates, dtype=torch.long)
    while total < examples:
        bs = min(256, examples - total)
        prompt, mask, _, target = batch(n_states, bs, depth, g)
        memory, aux = model.reason(
            prompt,
            prompt_attention_mask=mask,
            reasoning_steps=depth,
            return_diagnostics=True,
        )
        logits = model.direct_vocab_logits(memory[:, 0])
        pred = logits.argmax(dim=-1) - 4
        correct += int(pred.eq(target).sum())
        total += bs
        target_code = model.token.weight[4 + target]
        cosine.append(
            float(torch.nn.functional.cosine_similarity(memory[:, 0], target_code, dim=-1).mean())
        )
        for row in aux["history"]:
            probs = row["planner"]["machine"]["operator_probs"]
            generated_mass.append(float(probs[:, 2:].sum(dim=-1).mean()))
            selected = row["planner"]["machine"]["selected_operator"].cpu()
            selected_counts += torch.bincount(
                selected, minlength=selected_counts.numel()
            )
    labels = ["READ", "IDENTITY", "BLOCK_PRODUCT"] + [
        f"FLEX_{i}" for i in range(model.cfg.machine_operator_count)
    ]
    selected_total = int(selected_counts.sum())
    return {
        "accuracy": correct / total,
        "mean_target_cosine": sum(cosine) / len(cosine),
        "mean_nontrivial_operator_mass": sum(generated_mass) / max(len(generated_mass), 1),
        "selected_operator_fraction": {
            label: int(count) / max(selected_total, 1)
            for label, count in zip(labels, selected_counts.tolist(), strict=True)
        },
    }


def run(seed: int = 0, steps: int = 1000, n_states: int = 8) -> dict:
    model = make_model(seed, n_states)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4
    )
    g = torch.Generator().manual_seed(seed + 1000)
    history = []
    model.train()
    for step in range(1, steps + 1):
        depth = int(torch.randint(1, 4, (1,), generator=g))
        prompt, mask, answer, _ = batch(n_states, 128, depth, g)
        loss, aux = model.loss(
            prompt,
            answer,
            prompt_attention_mask=mask,
            reasoning_steps=depth,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.machine_cell.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "depth": depth, "loss": float(loss.detach())})
    model.eval()
    evaluation = {
        str(depth): evaluate(model, n_states, depth, seed + 5000 + depth)
        for depth in (1, 2, 3, 4, 6, 8)
    }
    return {"seed": seed, "steps": steps, "training": history, "evaluation": evaluation}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = run(args.seed, args.steps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
