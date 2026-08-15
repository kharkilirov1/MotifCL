# FOG-10M trained experimental candidate

## Status

**Exact one-hop token binding is validated; reasoning remains unvalidated.**
The primary bundled checkpoint is
`checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt` (SHA-256
`d9a5523adc85a049d970f51c2bd75c6f88c64a533d8614077334999a1f4c5960`).
It inherits lexical weights trained for 1,600 optimizer steps and 1,358,852
target tokens from 3,000 TinyStories, then trains one binding-sharpness scalar
for 40 steps. This scale and task remain insufficient for a useful
general-purpose language model.

The original 10,035,848-parameter TinyStories checkpoint remains bundled as a
secondary legacy artifact.

Bundled release weights are lossy BF16 inference exports of the FP32 training
states. They omit optimizer, scheduler, scaler, and RNG state.

Strict latent SFT was also attempted. It failed on held-out GSM8K and remained
weak on synthetic addition, so these checkpoints must not be described as a
working mathematical reasoner.

A separate `query_bound_v2` architecture now exists, but it must not be
confused with the lexically trained legacy release. Controlled exact-code
binding gates succeeded. The current **10,000,039-parameter** production path
also passed mapping-disjoint token lookup at 4032/4032 in all three seeds, with
all zero/deranged/query-cyclic controls at 0/4032. This validates one-step
binding at `R=1`, not general reasoning or arithmetic.

## Legacy geometry and v2 delta

| component | setting / unique parameters |
|---|---:|
| vocabulary / width | 8,192 / 320 |
| decoder blocks / heads / FFN | 4 / 5 / 1,344 |
| token embedding (tied head) | 2,621,440 |
| backbone | 5,258,176 |
| latent planner | 1,692,232 |
| persistent memory | 464,000 |
| **total** | **10,035,848** |

Latent geometry is `K=4` new slots, fixed `R=4` recurrent steps, compare rank
80, and `N=8` persistent slots. Memory grows `4 → 8`, then is learned-compressed
from 12 back to 8 on steps 3 and 4. `max_seq_len=512` counts lexical prompt,
latent memory, and answer prefix together.

The current v2 preset keeps `V=8192, d=320, L=4, H=5, FFN=1344, K=4,
rank=80`, uses an auxiliary planner width of 2,330, and contains
**10,000,039 unique parameters**. Its architecture contract is
`query_conditioned/direct_latent`, with one protected binding slot, a fixed-size
reusable four-slot workspace, and the single relative binding offset `(2,)`.
Runtime recurrent depth is configurable, but the positive 10M evidence uses
exactly `R=1`.

The primary v2 carrier is selected by Q/K address comparison. Payload
coordinates bypass learned V/O projection, the carrier bypasses auxiliary
workspace mixing, and reusable memory copies the current primary exactly rather
than appending/compressing it. The first output token is read by a cosine head
tied to the normalized token embedding codebook. Subsequent tokens use actual
prior answer tokens rather than a new blank BOS retrieval state.

Legacy-to-v2 migration copies only tensors with explicitly compatible
semantics. It is an initialization operation. In the successful lookup runs,
all migrated lexical weights and binding directions stayed frozen; only the
single address-softmax sharpness scalar was trained.

## Training data and checkpoints

- BPE: 8,192 tokens, trained on TinyStories `train[0:3000]`.
- Primary lexical checkpoint: 400 steps at `3e-4`, then 1,200 steps at
  `1.5e-4`, sequence length 128, batch size 4, FP32/CPU.
- Strict GSM8K: first 1,024 rows of `train`, deterministic 896/128 split,
  answer-only targets, memory-only decoder, 400 steps.
- Full-CoT diagnostic: the same 896/128 split, full prompt and full textual
  target, 400 steps.
- Addition diagnostic: all ordered pairs 0…19, stable pair-disjoint 320/80
  split, memory-only decoder, 800 steps. No held-out pair's reversed pair was
  present in train; the small number domain, operands, and templates were still
  shared, so this is not a broad arithmetic-OOD test.

The official GSM8K test split was not downloaded or used. Exact source hashes
are in `data_cache/*.manifest.json`.

## Evaluation

| checkpoint / task | result |
|---|---:|
| TinyStories, 400-step validation | loss 4.4758; PPL 87.86 |
| TinyStories, 1,600-step 10-batch selection | loss 3.7194; PPL 41.24; token acc. 32.93% |
| TinyStories, random → final full 300 rows | loss 9.0445 → 3.7529; PPL 8471.95 → 42.64 |
| strict GSM8K validation | 1/128 exact (0.78%) |
| strict GSM8K, zero/shuffled memory | 1/128 / 1/128 |
| full-CoT GSM8K validation | 0/128 exact |
| addition validation, normal/zero/shuffled | 6.25% / 5.00% / 1.25% exact |
| matched direct Transformer, locked test seeds 0/1/2 | 100.00% / 26.46% / 99.98% |
| matched FOG full, locked test seeds 0/1/2 | 12.28% / 12.35% / 12.28% |
| matched FOG strict, locked test seeds 0/1/2 | 12.28% / 12.28% / 12.28% |
| legacy frozen probes, query/proposal/memory | near 12.5% macro chance |
| raw query-addressed row, linear/MLP probes | 100.00% / 100.00% |
| binding-v2 exact-code locked test, seeds 0/1/2 | 4032/4032 each |
| four-digit binding locked test, seeds 0/1/2 | 4096/4096 exact each |
| current 10M-v2 token lookup validation, seeds 42/0/1 | 1024/1024 each |
| current 10M-v2 token lookup locked test, seeds 42/0/1 | 4032/4032 each |
| current 10M-v2 zero/target/query controls | 0/4032 each in every seed |
| current 10M-v2 address hit / mean correct mass | 100%; 98.42% / 96.62% / 99.83% |

The GSM8K strict model collapsed toward the frequent output `12`; zeroing or
shuffling memory did not change exact match. Full-CoT greedy output was
repetitive and malformed. Addition shows a small intervention gap, but 5/80 is
not task mastery. Teacher-forced token metrics include easy EOS and partial
digits and must not replace sequence exact match.

In the original matched structured lookup gate, the direct control had 69,184 trainable
parameters and FOG had 139,204. The mapping tables were hash-disjoint across
train/validation/test, shared initialization was name-stable, and the locked
test was evaluated only after protocol selection. Zeroing or target-deranged
shuffling FOG memory barely changed predictions. A one-pass new-BOS readout
control and a separate lossless two-pass control without planner compression
also stayed at chance after 1,000 steps. The evidence therefore points above
planner compression to the latent interface, not merely insufficient parameter
count.

Subsequent frozen probes refined the mechanism. Permutation-invariant row
pooling was mapping-blind, while exact and dot-product query-conditioned row
selection were 100% decodable. Linear, small-MLP, and learned slot-attention
probes did not recover an unseen-mapping target from legacy query hidden
states, proposals, or persistent memory. The legacy reader also ignored an
oracle one-hot answer payload until its reader side was adapted. The legacy
failure therefore contains both a writer/binding defect and a reader defect;
BOS alone was not a complete diagnosis.

The exact-code v2 gate freezes orthogonal key/value codebooks and ties the
classifier to the value codebook, preventing an arbitrary downstream decoder
from hiding coordinate corruption. All three saved checkpoints scored 100% on
4,032 locked-test mappings. Zeroing the primary carrier fell to the empirical
single-class baseline; target- and query-deranging scored zero. The four-digit
gate similarly scored 100% exact on 4,096 test tables in all seeds, with both
derangements at zero exact. These tests validate precise selection and storage,
not multi-step reasoning.

The production-scale token gate then exercised the migrated 8,192-token
embedding, four-layer backbone, query-conditioned writer, reusable `K=4`
workspace, checkpoint loading, and cosine tied direct head. Oracle copy decoded
8192/8192 vocabulary embeddings into themselves. Three independently migrated
model seeds trained only `planner.bind.logit_scale` for 40 steps and each
scored 100% on 1,024 validation examples followed by 4,032 locked-test
examples. Address argmax hit the correct key 100% of the time. The mean
locked-test mass on that address was 98.42%, 96.62%, and 99.83%.

Normal NLL remained 6.418/6.429/6.412 despite perfect top-1. Uniform NLL over
8,192 tokens is 9.01091. Because the lexical codebook, cosine head, and output
temperature were frozen, this gate establishes identity ranking but not
well-calibrated language probabilities.

## Known limits

- The lexical corpus is only 3,000 training stories and is repeatedly cycled.
- Lexical pretraining updates the shared embedding/backbone; it does not train
  the planner and persistent-memory path. Those modules are only adapted in the
  separate SFT diagnostic checkpoints.
- GSM8K SFT contains only 896 training examples and 10,510 answer target tokens.
- The matched direct baseline was not equally stable: one of three seeds only
  reached 26.46% on the locked test while the other two were nearly perfect.
- Of three earlier fixed-operator toy runs, one collapsed to chance.
- Fresh one-hop permutation lookup remained at chance for 1–3 backbone layers.
- Four latent slots had mean pairwise direction cosine near 0.9999 in toy
  checkpoints, so they are not established as independent parallel thoughts.
- A FOG forward is more expensive than a standard 10M decoder because the
  backbone runs once per latent step and once for answer decoding.
- Exact-code gates use synthetic mappings and frozen orthogonal codebooks; real
  token embeddings and natural-language relations are substantially harder.
- Selecting an already-present four-digit payload is not arithmetic: no carry,
  algorithm, or unseen numerical computation is required.
- Lookup does not establish compositional reasoning, useful recurrent depth,
  or independent parallel thoughts in auxiliary slots.
- The positive 10M token gate is a synthetic one-hop lookup with a prescribed
  `(2,)` address→payload offset. Only `R=1` and one trainable sharpness scalar
  were tested; `R>1`, relation composition, new-value computation, arithmetic,
  natural-language semantic binding, and GSM8K remain unvalidated.
- Perfect top-1 with NLL about 6.42 is not a confident or calibrated 8,192-way
  language prediction.

## Intended use

Use this package to reproduce the pipeline, continue pretraining, study the
latent-memory bottleneck, and compare architecture changes under matched data
and compute. Do not use it for factual assistance, mathematical answers, safety
critical decisions, or claims that hidden reasoning has been validated.

See `README_RU.md`, `TRAINING_REPORT_REAL_V2.md`, `EXPERIMENT_REPORT_V2.md`,
`MATCHED_EXPERIMENT_REPORT_RU.md`, `BINDING_V2_REPORT_RU.md`, and the JSON
artifacts for commands and raw evidence.

## Extended Pretraining (August 2026)

To further reduce loss and improve linguistic diversity, an extended pretraining cycle was initiated using a **1,000,000 sample dataset** mixed from TinyStories (500,000 samples) and FineWeb-Edu (500,000 samples). This expansion provides the model with a broader range of narrative structures and educational content, facilitating a more robust latent representation of natural language.

A critical technical improvement was the implementation of a **buffer-based streaming local record factory**. This allows the training pipeline to handle million-scale datasets on hardware with limited memory, such as the user's Xeon E5 and 8GB RAM configuration, by avoiding the preloading of the entire dataset into RAM. The training utilizes the `serious` recipe, targeting **20,000 steps** with a batch size of 2 and gradient accumulation of 16, resulting in an effective batch size of 32. The process was successfully resumed from step 9,000, and all scripts, improvements, and updated model cards have been synchronized with the GitHub repository.
