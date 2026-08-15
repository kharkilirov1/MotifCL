import os
import sys
import subprocess
from pathlib import Path
from tokenizers import Tokenizer

def main():
    os.system("color" if os.name == "nt" else "")
    
    tokenizer_path = Path("ports/rx580/fog/tinystories_3k_bpe.json")
    checkpoint_path = Path("checkpoints/fog_v3_rx580_lexical_30k.mclp")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/fog_v3_rx580_reasoning_sft.mclp")
    
    exe_path = Path("build/rx580-release/examples/cpp/13_fog_v3_inference.exe")
    demo_exe = Path("build/rx580-release/examples/cpp/17_fog_v3_ghost_room_test.exe")

    if not tokenizer_path.exists() or not exe_path.exists():
        print(f"Error: Missing tokenizer or compiled binary at {exe_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print("=" * 70)
    print("      FOG v3 (Latent Register Machine) — Interactive Chat & Logic")
    print("      Устройство: AMD Radeon RX 580 (Vulkan Compute Engine)")
    print("=" * 70)
    print("Команды:")
    print("  • Введите любой текст/начало фразы для генерации (на английском)")
    print("  • /temp <0.1-1.5>  - изменить креативность (текущая: 0.7)")
    print("  • /tokens <N>      - длина ответа в токенах (текущая: 60)")
    print("  • /ghost           - запустить стресс-тест 'Призрак в коридоре с шумом'")
    print("  • exit / quit      - выход")
    print("=" * 70)

    temperature = 0.7
    max_tokens = 60
    top_k = 40

    env = {
        **os.environ,
        "MOTIFCL_REQUIRE_VULKAN_COMPUTE": "1",
        "MOTIFCL_REQUIRE_VULKAN_MATMUL": "1",
        "MOTIFCL_REQUIRE_VULKAN_ATTENTION": "1",
    }

    while True:
        try:
            prompt = input("\n[FOG-v3 RX580] >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из чата.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", "q", "выход"):
            print("Сессия завершена.")
            break

        if prompt.startswith("/temp"):
            parts = prompt.split()
            if len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"Температура установлена: {temperature}")
                except ValueError:
                    print("Ошибка: укажите число, например /temp 0.6")
            continue

        if prompt.startswith("/tokens"):
            parts = prompt.split()
            if len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"Макс. токенов установлено: {max_tokens}")
                except ValueError:
                    print("Ошибка: укажите целое число, например /tokens 80")
            continue

        if prompt == "/ghost":
            print("\nЗапуск стресс-теста 'Призрак в коридоре' на RX 580...")
            subprocess.run([str(demo_exe), str(checkpoint_path)], env=env)
            continue

        # Normal text generation
        encoding = tokenizer.encode(prompt)
        prompt_ids = encoding.ids
        if not prompt_ids:
            prompt_ids = [1]
        prompt_csv = ",".join(str(x) for x in prompt_ids)

        cmd = [
            str(exe_path),
            "--checkpoint", str(checkpoint_path),
            "--prompt-ids", prompt_csv,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--top-k", str(top_k),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print(f"Ошибка инференса: {res.stderr}", file=sys.stderr)
            continue

        output_line = ""
        for line in res.stdout.splitlines():
            if line.startswith("OUTPUT_IDS:"):
                output_line = line[len("OUTPUT_IDS:"):].strip()
                break

        if not output_line:
            print("Ошибка декодирования вывода.")
            continue

        all_ids = [int(x) for x in output_line.split(",") if x]
        decoded_text = tokenizer.decode(all_ids)

        print("\n" + "-" * 50)
        print(decoded_text)
        print("-" * 50)

if __name__ == "__main__":
    main()
