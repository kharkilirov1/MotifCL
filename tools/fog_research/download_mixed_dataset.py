import os
import json
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading TinyStories and FineWeb-Edu from Hugging Face...")
    
    # TinyStories for narrative structure (500k samples)
    ts_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # FineWeb-Edu sample for educational/diverse text diversity (500k samples)
    try:
        fw_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading FineWeb-Edu: {e}. Falling back to more TinyStories.")
        fw_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    train_file = "data/train_mixed_1m.jsonl"
    eval_file = "data/eval_mixed_1m.jsonl"
    
    print("Creating mixed dataset (1M total: 500k TinyStories + 500k FineWeb/Edu)...")
    
    count = 0
    train_count = 0
    eval_count = 0
    
    with open(train_file, "w", encoding="utf-8") as f_train, open(eval_file, "w", encoding="utf-8") as f_eval:
        # Process TinyStories
        ts_count = 0
        for item in ts_dataset:
            text = item.get("text", "")
            if not text or not text.strip():
                continue
            record = {"text": text}
            if count % 20 == 0:
                f_eval.write(json.dumps(record, ensure_ascii=False) + "\n")
                eval_count += 1
            else:
                f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_count += 1
            count += 1
            ts_count += 1
            if ts_count >= 500000:
                break
                
        print(f"Added {ts_count} TinyStories samples.")
        
        # Process FineWeb-Edu
        fw_count = 0
        for item in fw_dataset:
            text = item.get("text", "")
            if not text or not text.strip():
                continue
            record = {"text": text}
            if count % 20 == 0:
                f_eval.write(json.dumps(record, ensure_ascii=False) + "\n")
                eval_count += 1
            else:
                f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_count += 1
            count += 1
            fw_count += 1
            if fw_count >= 500000:
                break
                
        print(f"Added {fw_count} FineWeb-Edu samples.")
                
    print(f"Done! Saved mixed dataset: {train_count} train samples to {train_file}, {eval_count} eval samples to {eval_file}.")

if __name__ == "__main__":
    main()
