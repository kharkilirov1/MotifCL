# FOG v3 (Finite Operator Grammar) — Research, Theory, and Bare-Metal Vulkan Implementation

## 1. Overview and Key Results

FOG v3 is a **neuro-symbolic latent register machine** integrated into the Transformer forward stack. It combines the expressive power of a Causal Language Model with the exact algebraic correctness of discrete typed register machines.

### Headline Empirical Results on AMD Radeon RX 580 (Vulkan)
- **Pretraining**: 30.72M tokens trained from scratch on a single RX 580 8GB using native Vulkan FP32 compute.
- **Stage 2 Reasoning SFT**: Multi-step state tracking curriculum ($R=1 \rightarrow R=2 \rightarrow R=4$), loss reduced from $18.75$ to $2.90 \times 10^{-5}$.
- **Depth Generalization ($R=1..12$)**: **100.0% accuracy** on 400 independent trials up to $3\times$ deeper than training ($R=12$). Pure 4-layer Transformer achieves **~8.0%** (chance level).
- **Ghost Room & Distractor Noise Intervention**: Injected **150+ random noise tokens** into context. FOG v3 maintains **100.0% accuracy**, while Soft-Attention drops to **8.0%**.
- **End-to-End Inference**: Sub-millisecond latency per logic transition (~100 microseconds/tick) with natural language story completion.

---

## 2. Theoretical Foundation (`research/fog/`)

The theoretical basis of FOG was developed across 37 controlled experiments (EXP-001 through EXP-037):

1. **Exact Address Binding (EXP-001 & `query_bound_v2`)**: Query-conditioned cosine address binder protects payload values from mixing with context tokens.
2. **Harmonic State Codebook & Unitary Rotation (EXP-003 – EXP-005)**: Continuous state representations in a Fourier/harmonic codebook ($amp = \sqrt{2}$, frequencies $h=1..d/2$) allow closed group operations without vector degradation.
3. **Finite Operator Grammar (`BLOCK_PRODUCT`)**: Recurrent state evolution via bilinear operators operating on typed registers (`Value`, `Control`, `Scratch0`, `Scratch1`).
4. **Structural Operator Compiler (EXP-023 – EXP-035)**:
   - *Eigengauge recovery*: Discovers canonical coordinates from dense hidden states.
   - *Commutant denoising*: Recovers shared 2x2 grammars from noisy activations.
   - *Black-box exploration*: Extracts discrete context graphs and local Jacobians from arbitrary external neural networks without retraining.

---

## 3. Architecture & Implementation in MotifCL

### Core Components:
- **`FogAddressBinderV3`** ([`src/nn/fog_v3.cpp`](file:///C:/Users/Kharki/Desktop/motifcl_production/src/nn/fog_v3.cpp)): Addresses memory slots and extracts query-conditioned values.
- **`FogRegisterCellV3`** ([`src/nn/fog_v3.cpp`](file:///C:/Users/Kharki/Desktop/motifcl_production/src/nn/fog_v3.cpp)): 4 physical registers updated per tick via discrete operator gates.
- **`ModernTransformerBlock`** ([`src/nn/transformer.cpp`](file:///C:/Users/Kharki/Desktop/motifcl_production/src/nn/transformer.cpp)): 4-layer Lexical Backbone trained on Vulkan.

### C++ Examples & Verification Suite:
- `09_fog_v3_rx580_pretrain.cpp`: Bare-metal pretraining on Vulkan.
- `11_fog_v3_structured_machine_gate.cpp`: Strict parameter contract & JVP verification.
- `12_fog_v3_reasoning_sft.cpp`: Staged SFT training loop.
- `13_fog_v3_inference.cpp`: High-speed token generator with Top-K/temperature sampling.
- `14_fog_v3_logic_demo.cpp`: Microsecond register trace and operator decoding.
- `15_fog_v3_e2e_prompt.cpp`: Real-world scenarios (Key tracking, Security base, Drone navigation).
- `16_fog_v3_ablation_test.cpp`: 400-trial blind benchmark comparing Pure Transformer vs FOG.
- `17_fog_v3_ghost_room_test.cpp`: Ghost Room Intervention with 150+ noise distractor tokens.
- `18_fog_v3_fairy_tale_logic.cpp`: End-to-end fairy-tale story generation with register logic.

---

## 4. Integration into Existing Models (Grafting & Sidecar)

FOG v3 is designed to be grafted onto large external models (LLaMA-3, Qwen-2.5, Mistral, Gemma, Phi-3) via:
1. **GGUF / HuggingFace Loader** ([`include/motifcl/gguf.hpp`](file:///C:/Users/Kharki/Desktop/motifcl_production/include/motifcl/gguf.hpp), [`include/motifcl/nn/hf_compat.hpp`](file:///C:/Users/Kharki/Desktop/motifcl_production/include/motifcl/nn/hf_compat.hpp)).
2. **Canonical Gauge Alignment (EXP-029)**: Zero-parameter alignment of external hidden states $h \in \mathbb{R}^d$ to the canonical FOG coordinate system.
3. **Sidecar Coprocessor**: A 4-layer hybrid stack where the external LLM handles language representation while FOG registers execute exact discrete multi-step transitions.
