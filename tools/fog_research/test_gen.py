import torch
from fog_lmw.model import FOGLatentReasoner
from fog_lmw.config import FOGReasonerConfig
from tokenizers import Tokenizer
import torch.nn.functional as F

def main():
    tokenizer = Tokenizer.from_file("tokenizer/tinystories_3k_bpe.json")
    checkpoint_path = "checkpoints/pretrain_large/best.pt"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = FOGReasonerConfig(**payload["model_config"])
    model = FOGLatentReasoner(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    
    bos_token_id = tokenizer.token_to_id("<|endoftext|>") # Usually 0 or 1, manifest says 1 acts as BOS
    # In RELEASE_MANIFEST, it says vocab_size 8192, tinystories_3k_bpe.
    # Let's check the manifest again.
    
    prompts = [
        "Once upon a time, there was a little girl named Lily.",
        "The big blue bird flew over the",
        "Tom wanted to play with his"
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        tokens = tokenizer.encode(p).ids
        prompt_ids = torch.tensor([tokens], device='cpu')
        
        with torch.no_grad():
            generated_ids, _ = model.generate(
                prompt_ids=prompt_ids,
                bos_token_id=1, # From demo.py and manifest hint
                max_new_tokens=30
            )
        
        result = tokenizer.decode(generated_ids[0].tolist())
        print(f"Generated: {result}")

if __name__ == "__main__":
    main()
