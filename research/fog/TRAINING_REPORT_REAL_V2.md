# Real-data training report: FOG latent reasoner, 10M

Date: 2026-08-12  
Status: training completed on CPU; the pipeline works, language-model pretraining learned, but reliable reasoning did not emerge at this scale.

## Executive result

The exact model has **10,035,848 unique parameters**. It was trained for 1,600 TinyStories steps (1,358,852 target tokens), then tested with three supervised regimes: strict latent-only GSM8K answers, full-prompt GSM8K chain-of-thought, and a controlled addition task.

The strongest clean result is lexical pretraining. On a fresh evaluation over all 300 local TinyStories validation rows (63,668 target tokens), loss fell from **9.0445 to 3.7529** and perplexity from **8471.95 to 42.64**. Greedy samples contain local English/TinyStories-like syntax, but remain strongly repetitive and semantically inconsistent.

The reasoning result is negative. Strict GSM8K achieved **1/128 exact** and was unchanged by zeroing or shuffling memory; it mostly emitted `12`. Full chain-of-thought GSM8K achieved **0/128 exact** with malformed repetitive generations. Addition achieved **5/80 exact (6.25%)**, only one example above the 5% majority-answer baseline. The memory-shuffle drop is a weak indication that latent content affects predictions, not evidence that the model learned addition.

Pretraining and reasoning should not be conflated: causal-LM pretraining exercised the shared embeddings/backbone/LM head; the planner and latent-memory route were inactive until SFT.

## Reproducible model contract

| Field | Value |
|---|---:|
| Unique parameters | 10,035,848 |
| Vocabulary | 8,192 |
| Width / layers / heads | 320 / 4 / 5 |
| FFN width | 1,344 |
| Maximum sequence | 512 |
| Latent slots / memory slots | 4 / 8 |
| Reasoning steps | 4 |
| Compare rank | 80 |
| Reasoning modes | 8 |

All recorded training used PyTorch 2.9.1+cpu, FP32, seed 42, on an Intel Xeon Platinum 8370C host. Recorded evaluations used four CPU threads.

Tokenizer: 8,192-token BPE trained on the 3,000-row local TinyStories subset, minimum frequency 2.

```text
tokenizer/tinystories_3k_bpe.json
SHA-256 a4d8b2413791f5b5a93653720ef545f63ef35112a449b7e8bccbdb08b463af61
```

## Data and provenance

The real subsets were fetched on 2026-08-12 through the Hugging Face Dataset Viewer `/rows` API. That API response is not revision-pinned. Reproducibility therefore depends on the included local content SHA-256 and manifest, not on the repository revision constants used by `train_real.py` when loading directly from Hugging Face.

| Local subset | Source | Rows | Source rows | SHA-256 |
|---|---|---:|---:|---|
| TinyStories train | `roneneldan/TinyStories`, `default/train` | 3,000 | 0–2,999 | `445c316f3d0868156ce2c910c924edc7534866bfc22b8b9f8437deed9ecf5fce` |
| TinyStories validation | `roneneldan/TinyStories`, `default/validation` | 300 | 0–299 | `e1412321b5d5c970111bad3fa5f79598a411617f0da58f5815f1b56f96c8a04a` |
| GSM8K train subset | `openai/gsm8k`, `main/train` | 1,024 | 0–1,023 | `bfa19819db9f14ee2e63b66351b1a8a189c9c8ce075fa29c484d47053775057f` |

The downloader rejects partial pages, truncated cells, discontinuous indices and row-count mismatches. The downloaded rows were contiguous and unique; TinyStories had no blank text and all 1,024 GSM8K records had the `####` final-answer marker. **The official GSM8K test split was not accessed.** GSM8K train/validation below means a deterministic seed-42 split of these 1,024 official train rows.

Addition is locally generated over every ordered pair `a,b ∈ [0,19]` with four templates. The rule `(a*53 + b*97) % 5 == 0` gives 320 train and 80 validation examples. Exact ordered pairs are disjoint, and an explicit check found zero validation pairs whose reversed `(b,a)` pair occurs in train.

| Addition split | Rows | SHA-256 |
|---|---:|---|
| Train | 320 | `31bdf3da7b103b3caf2e315d980ecdf9dab23e73613d90d057ada02c6a7b89d6` |
| Validation | 80 | `1cf6765667d3e415def4e87dd14aff41ae270a8fa752dba9d873f843c7cda687` |

## Experiment 1: TinyStories pretraining

Two consecutive CPU stages used batch size 4, sequence length 128, one optimizer step per batch, cosine decay, 20 warmup steps and weight decay 0.1.

| Stage | LR | Steps | Target tokens | Bounded validation loss | Bounded PPL |
|---|---:|---:|---:|---:|---:|
| 1 | 3e-4 | 400 | 339,043 | 4.4758 | 87.86 |
| 2 | 1.5e-4 | +1,200 | +1,019,809 | 3.7194 | 41.24 |
| Combined | — | 1,600 | 1,358,852 | — | — |

The checkpoint-selection values above use ten validation batches. A separate full-subset reevaluation provides the primary comparable result:

| Checkpoint | Step | Loss | PPL | Token accuracy |
|---|---:|---:|---:|---:|
| Random initialization | 0 | 9.044516 | 8471.953 | 0.0016% |
| Stage 1 | 400 | 4.503845 | 90.364 | 24.3702% |
| Final | 1,600 | 3.752902 | 42.645 | 32.9553% |

Primary trained checkpoint:

```text
checkpoints/fog_10m_tinystories_1_36m_pretrained.pt
SHA-256 ee193e6e0fc434c24343ff6601e59acf20de0565e08bdcce37763e5d6f45bcab
```

The release archive carries smaller portable BF16 inference exports. These are intentionally lossy relative to the FP32 sources; each payload records its source filename and SHA-256:

| Release checkpoint | Purpose | SHA-256 |
|---|---|---|
| `fog_10m_tinystories_1_36m_pretrained_bf16.pt` | continuation base | `0cc9650eb2c1de0109fa517ebf5f22589f1b24edd7e8818e522cbd64afde0e7b` |
| `fog_10m_gsm8k_896_best_bf16.pt` | strict GSM8K diagnostic | `27f535b3303a912282ec7cba83601f1f6f51a5cc5d0573cae0292198c855d513` |
| `fog_10m_addition_strict_best_bf16.pt` | strict addition diagnostic | `df59c85d1c860aa4fc1200a09b80cc8f219da8b26fa86b610d534a3324b4fe8d` |

The FP32 hash above identifies the exact source checkpoint used for the reported metrics; FP32 source weights may be omitted from the compact release archive. Use the pretrained BF16 file in the first row as the bundled continuation base.

Representative commands:

```bash
python train_real.py tokenizer \
  --local-data data_cache/tinystories_train_3000.jsonl \
  --text-field text --max-samples 3000 --vocab-size 8192 \
  --min-frequency 2 --seed 42 \
  --output tokenizer/tinystories_3k_bpe.json

python train_real.py pretrain \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint checkpoints/fog_10m_tinystories_3k_init.pt \
  --local-data data_cache/tinystories_train_3000.jsonl \
  --local-eval-data data_cache/tinystories_validation_300.jsonl \
  --text-field text --sequence-length 128 --batch-size 4 \
  --max-steps 400 --warmup-steps 20 --lr 3e-4 \
  --weight-decay 0.1 --eval-every 100 --eval-batches 10 \
  --save-every 100 --seed 42 --device cpu --precision fp32 \
  --checkpoint-dir /tmp/fog10m_real_pretrain

python train_real.py pretrain \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint checkpoints/fog_10m_tinystories_3k_pretrained.pt \
  --local-data data_cache/tinystories_train_3000.jsonl \
  --local-eval-data data_cache/tinystories_validation_300.jsonl \
  --text-field text --sequence-length 128 --batch-size 4 \
  --max-steps 1200 --warmup-steps 20 --lr 1.5e-4 \
  --weight-decay 0.1 --eval-every 200 --eval-batches 10 \
  --save-every 200 --seed 42 --device cpu --precision fp32 \
  --checkpoint-dir /tmp/fog10m_real_pretrain_stage2
```

## Experiment 2: strict latent-only GSM8K

The decoder received only BOS plus latent memory; the lexical question was not passed to it. Target mode was the numeric final answer. The 896/128 run started from the 400-step TinyStories checkpoint and used batch 8, LR 3e-4, 20 warmup steps, 400 steps and 10,510 target tokens.

Best teacher-forced validation at step 400 was loss 1.73795, perplexity 5.68568, token accuracy 40.14%. This token metric includes short answer subwords and EOS, so greedy sequence exact match is decisive:

| Split | Normal | Zero memory | Shuffled memory |
|---|---:|---:|---:|
| Train sample, 128 | 2/128 (1.56%) | 2/128 | 2/128 |
| Held-out, 128 | 1/128 (0.78%) | 1/128 | 1/128 |

The model mostly generated `12`. Identical intervention scores show no useful latent-memory dependence for this run.

An overfit gate on 64 train / 16 held-out rows used 400 steps. At the last step it reached 8/64 exact normally, 4/64 with zero memory and 2/64 with shuffled memory. This establishes some input-specific memory use on seen examples, but memorization was weak; the best-validation checkpoint scored 0/16 under all three interventions.

Representative strict command:

```bash
python train_real.py sft \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --init-checkpoint checkpoints/fog_10m_tinystories_3k_pretrained.pt \
  --local-data data_cache/gsm8k_train_1024.jsonl \
  --validation-size 128 --batch-size 8 --max-steps 400 \
  --warmup-steps 20 --lr 3e-4 --weight-decay 0.01 \
  --prompt-field question --response-field answer \
  --prompt-template $'Question: {prompt}\nAnswer:' \
  --target-mode final --decoder-mode memory-only \
  --max-prompt-length 128 --max-answer-length 12 \
  --reasoning-steps 4 --eval-every 100 --eval-batches 16 \
  --save-every 100 --seed 42 --device cpu --precision fp32 \
  --checkpoint-dir /tmp/fog10m_gsm8k_896
```

## Experiment 3: full-prompt, full-chain GSM8K

This diagnostic started from the stronger 1.36M-token checkpoint and exposed the full question to the decoder. It trained on complete GSM8K responses for 400 steps, batch 4, LR 2e-4 and 178,227 target tokens.

Teacher-forced validation improved through losses 4.1273, 3.7127, 3.5647 and 3.4921, ending at 31.85% token accuracy. Yet 96-token greedy generation scored **0/128 exact** and repeatedly produced malformed sequences such as runs of angle brackets and repeated `12=12`. Thus this route learned surface continuation statistics without usable GSM8K solutions.

## Experiment 4: strict addition gate

This run started from the stronger TinyStories checkpoint and used memory-only decoding, final-answer targets, batch 16, LR 3e-4, 800 steps and 36,640 target tokens.

Teacher-forced validation loss ended at 0.82493 with 68.56% token accuracy. Again, short digit tokens and EOS inflate this metric relative to sequence-level correctness.

| Split | Normal | Zero memory | Shuffled memory | Majority baseline |
|---|---:|---:|---:|---:|
| Train, 320 | 24/320 (7.50%) | 12/320 (3.75%) | 8/320 (2.50%) | 16/320 (5.00%) |
| Held-out, 80 | 5/80 (6.25%) | 4/80 (5.00%) | 1/80 (1.25%) | 4/80 (5.00%) |

The held-out normal score beats the majority baseline by only **one example / 1.25 percentage points**. The intervention gap says the prediction changes when memory content is destroyed, but the absolute result does not establish arithmetic ability.

## What was learned—and what was not

What worked:

- Exact 10M configuration trains end-to-end without numerical failure.
- Portable checkpoints load with tokenizer/config/hash metadata.
- TinyStories validation improved dramatically and monotonically with more tokens.
- Strict controls can detect when per-example latent memory affects predictions.

What did not work:

- No tested run learned reliable GSM8K reasoning.
- Full-chain generation was repetitive and malformed.
- Addition exact match remained effectively at the frequency baseline.
- Teacher-forced token accuracy overstated practical performance on short numeric targets.

The honest conclusion is therefore: **this is a trained, runnable research checkpoint, not a capable 10M reasoning model**. The evidence supports extending lexical pretraining substantially (tens to hundreds of millions of tokens), pretraining or curriculum-training the planner/memory path, adding matched no-latent baselines, evaluating multiple seeds, and improving the latent interface before making a reasoning claim.

## Machine-readable evidence

The consolidated exact values, hashes and artifact paths are in `artifacts/real_training/metrics.json`. Raw generation/intervention outputs remain in the same directory. Dataset manifests are adjacent to their local JSONL files.
