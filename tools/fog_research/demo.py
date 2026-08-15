import torch
from fog_lmw import FOGReasonerConfig, FOGLatentReasoner


def main():
    torch.manual_seed(0)
    cfg = FOGReasonerConfig(
        vocab_size=1024,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=384,
        max_seq_len=256,
        latent_slots=8,
        reasoning_steps=4,
        compare_rank=32,
        planner_ff=256,
        # 8 -> 16 -> learned compression back to 16 on steps 3 and 4.
        memory_slots=16,
        n_reasoning_modes=4,
    )
    model = FOGLatentReasoner(cfg)

    prompt = torch.randint(4, cfg.vocab_size, (2, 12))
    # token 1 acts as BOS in this toy example
    answer = torch.cat(
        [torch.ones(2, 1, dtype=torch.long), torch.randint(4, cfg.vocab_size, (2, 7))],
        dim=1,
    )

    loss, aux = model.loss(prompt, answer, return_diagnostics=True)
    print("loss:", loss.detach().item())
    print("memory sizes:", [s["memory_size"] for s in aux["history"]])
    print("final latent shape:", tuple(aux["history"][-1]["latent"].shape))
    print(
        "compression active:",
        [s["memory"]["compressed"] for s in aux["history"]],
    )


if __name__ == "__main__":
    main()
