import os
import json
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading roneneldan/TinyStories from Hugging Face (500k target)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    train_file = "data/train.jsonl"
    eval_file = "data/eval.jsonl"
    
    print("Downloading and processing 500,000 samples...")
    count = 0
    train_count = 0
    eval_count = 0
    
    with open(train_file, "w", encoding="utf-8") as f_train, open(eval_file, "w", encoding="utf-8") as f_eval:
        for item in dataset:
            text = item.get("text", "")
            if not text.strip():
                continue
            
            record = {"text": text}
            # 95% train, 5% eval
            if count % 20 == 0:
                f_eval.write(json.dumps(record, ensure_ascii=False) + "\n")
                eval_count += 1
            else:
                f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_count += 1
                
            count += 1
            if count % 50000 == 0:
                print(f"Processed {count} samples...")
            if count >= 500000:
                break
                
    print(f"Done! Saved {train_count} train samples to {train_file} and {eval_count} eval samples to {eval_file}.")

if __name__ == "__main__":
    main()
