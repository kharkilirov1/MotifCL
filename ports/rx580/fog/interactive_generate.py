import argparse
import subprocess
import sys
from pathlib import Path
from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="FOG v3 Interactive Text Generation on RX 580 Vulkan")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fog_v3_rx580_reasoning_sft.mclp", help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default="ports/rx580/fog/tinystories_3k_bpe.json", help="Tokenizer json path")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling")
    args = parser.parse_args()

    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tok_path))
    encoding = tokenizer.encode(args.prompt)
    prompt_ids = encoding.ids

    if not prompt_ids:
        prompt_ids = [1] # BOS if empty

    prompt_csv = ",".join(str(i) for i in prompt_ids)
    exe_path = Path("build/rx580-release/examples/cpp/13_fog_v3_inference.exe")

    if not exe_path.exists():
        print(f"Error: Inference executable not found at {exe_path}. Build it first!", file=sys.stderr)
        sys.exit(1)

    cmd = [
        str(exe_path),
        "--checkpoint", args.checkpoint,
        "--prompt-ids", prompt_csv,
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
    ]

    env = {
        **subprocess.os.environ,
        "MOTIFCL_REQUIRE_VULKAN_COMPUTE": "1",
        "MOTIFCL_REQUIRE_VULKAN_MATMUL": "1",
        "MOTIFCL_REQUIRE_VULKAN_ATTENTION": "1",
    }

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"Error executing C++ inference binary:\n{res.stderr}\n{res.stdout}", file=sys.stderr)
        sys.exit(res.returncode)

    output_line = ""
    for line in res.stdout.splitlines():
        if line.startswith("OUTPUT_IDS:"):
            output_line = line[len("OUTPUT_IDS:"):].strip()
            break

    if not output_line:
        print(f"Could not parse OUTPUT_IDS from:\n{res.stdout}", file=sys.stderr)
        sys.exit(1)

    all_ids = [int(x) for x in output_line.split(",") if x]
    decoded_text = tokenizer.decode(all_ids)
    
    print("\n" + "=" * 60)
    print(f"PROMPT:    {args.prompt}")
    print(f"GENERATED: {decoded_text}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
