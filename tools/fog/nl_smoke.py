# -*- coding: utf-8 -*-
"""Natural-language smoke test for FOG v3 checkpoints on real RX 580 (Vulkan)."""
import os, sys, subprocess, time
from pathlib import Path
from tokenizers import Tokenizer

WD = Path(r"C:\Users\Kharki\Desktop\motifcl_production")
TOK = WD / "ports/rx580/fog/tinystories_3k_bpe.json"
EXE = WD / "build/rx580-release/examples/cpp/13_fog_v3_inference.exe"
CKPTS = [
    "checkpoints/fog_v3_rx580_lexical_30k.mclp",   # pure lexical backbone
    "checkpoints/fog_v3_rx580_reasoning_sft.mclp", # SFT (state codebook in 4..19)
]
PROMPTS = [
    "Once upon a time there was a little girl named Lily.",
    "The little boy went to the park and",
    "Once upon a time, a brave knight",
    "Lily loved to play with her friends.",
]
MAX_TOKENS = 60
TEMP = 0.7
TOP_K = 40

env = {**os.environ,
       "PYTHONUTF8": "1",
       "MOTIFCL_REQUIRE_VULKAN_COMPUTE": "1",
       "MOTIFCL_REQUIRE_VULKAN_MATMUL": "1",
       "MOTIFCL_REQUIRE_VULKAN_ATTENTION": "1"}

tok = Tokenizer.from_file(str(TOK))
print(f"tokenizer vocab={tok.get_vocab_size()}")

def decode(ids):
    out = tok.decode(ids)
    # `decode` already turns Ġ into spaces. Show raw too for shape.
    return out.replace("<|endoftext|>", "<EOS>").strip()

# sanity: report what tokens 4..19 decode to (these slots were overwritten in SFT)
print("tokens 4..19 (overwritten by state codebook in SFT):")
for i in range(4, 20):
    try:
        w = tok.decode([i])
    except Exception:
        w = "?"
    print(f"  tok[{i}] = {w!r}")

print("\n" + "="*78)
for ck in CKPTS:
    ckp = WD / ck
    print(f"\n### CHECKPOINT: {ck}")
    print("-"*78)
    if not ckp.exists():
        print("  (missing, skip)")
        continue
    for prompt in PROMPTS:
        enc = tok.encode(prompt)
        ids = enc.ids
        csv = ",".join(str(x) for x in ids)
        cmd = [str(EXE), "--checkpoint", str(ck),
               "--prompt-ids", csv,
               "--max-tokens", str(MAX_TOKENS),
               "--temperature", str(TEMP), "--top-k", str(TOP_K)]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               cwd=str(WD), timeout=120, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            print(f"\n[timeout] prompt={prompt!r}")
            continue
        dt = time.time() - t0
        line = ""
        for ln in r.stdout.splitlines():
            if ln.startswith("OUTPUT_IDS:"):
                line = ln[len("OUTPUT_IDS:"):]; break
        all_ids = [int(x) for x in line.split(",") if x.strip()] if line else []
        gen_ids = all_ids[len(ids):]
        out = tok.decode(gen_ids) if gen_ids else "(no output)"
        print(f"\n[seed] {prompt}")
        print(f"  ids={ids}  ({dt:.1f}s, exit={r.returncode})")
        if r.returncode != 0 and r.stderr:
            print(f"  stderr: {r.stderr.strip()[:300]}")
        print(f"  -> {out!r}")
print("\n" + "="*78)
print("done")