# DiffUME — Architecture & Developer Reference

**DiffUME** (Diffusion Unified Multimodal Embeddings) converts Qwen2.5-VL from
a conventional autoregressive VLM into a **block-wise discrete diffusion
embedding model**.  It re-implements the SDAR-VL training recipe
(Cheng et al., arXiv:2512.14068) — ABNS + EMRS + PBNC — on top of the
UME-R1 multimodal embedding corpus, producing paired (discriminative, generative)
embedding vectors from a shared forward pass.

---

## Table of contents

1. [Motivation & design decisions](#1-motivation--design-decisions)
2. [Repository layout](#2-repository-layout)
3. [BD3 background](#3-bd3-background)
4. [Architecture overview](#4-architecture-overview)
5. [Module reference](#5-module-reference)
   - 5.1 [Special tokens — `diffume/tokens.py`](#51-special-tokens--diffumetokenspy)
   - 5.2 [Data pipeline](#52-data-pipeline)
   - 5.3 [Diffusion core](#53-diffusion-core)
   - 5.4 [Models](#54-models)
   - 5.5 [Losses](#55-losses)
   - 5.6 [Training](#56-training)
   - 5.7 [Inference](#57-inference)
   - 5.8 [Evaluation](#58-evaluation)
6. [Training details](#6-training-details)
7. [Configs & scripts](#7-configs--scripts)
8. [Tests](#8-tests)
9. [Known deviations from the plan](#9-known-deviations-from-the-plan)
10. [Glossary](#10-glossary)

---

## 1. Motivation & design decisions

| Decision | Value | Rationale |
|---|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` | UME-R1 default; tested multimodal decoder |
| Conversion method | SDAR-VL BD3 recipe (re-implemented; no SDAR-VL weights exist) | Block-causal attention + masked diffusion objective |
| Training corpus | `umer1_sft_processed` (1.46 M paired rows) | Reasoning spans pre-filled (`<think>…</think><answer>…</answer><gen_emb>`) |
| Embedding tokens | `<gen_emb>` (id 151666), `<disc_emb>` (id 151667), `<\|mask\|>` (id 151665) | UME-R1 dual-mode; +1 diffusion mask token |
| Attention mask | Block-causal additive float mask (0 / -∞), passed as 4-D tensor | Qwen2.5-VL accepts 4-D masks and short-circuits its own causal-mask builder |
| Parallelism | FSDP `SHARD_GRAD_OP` (ZeRO-2 equivalent), 8 × A100-40 GB | DeepSpeed not available on GPU nodes at runtime; FSDP is PyTorch-native |
| Block size | 32 tokens | Paper default for the reasoning regime; 32 × bidirectional context |
| Losses | `λ_diff · (L_diff_q + L_diff_p) + λ_disc · ClipLoss + λ_gen · ClipLoss` | λ = 1.0 for all terms in v0 |

**Key design choice — no vendored modeling file.** The plan originally called for
forking `modeling_qwen2_5_vl.py`.  Instead, we pass an externally-built 4-D
additive mask directly to `Qwen2_5_VLModel.forward()`.  Transformers ≥ 4.45
detects a 4-D mask and returns it unchanged from `_update_causal_mask`, so
the block-causal pattern is respected without any copy of the model source.

**Key FSDP fix.** FSDP intercepts parameter-gathering hooks on `nn.Module.forward()`,
not on custom methods.  `DiffumeModel.forward()` is a thin wrapper that calls
`forward_train(**kwargs)`; the trainer calls `model(...)` (not
`model.forward_train(...)`) so FSDP gathers sharded parameters before each forward.

---

## 2. Repository layout

```
DiffUME/
├── pyproject.toml                   # standalone pip-installable package
├── requirements-diffume.txt
├── configs/
│   ├── diffume_debug_1gpu.yaml      # 5-step smoke config
│   ├── diffume_stage1.yaml          # full 8-GPU SFT config (30 000 steps)
│   └── accelerate/
│       ├── single_gpu.yaml          # 1-GPU accelerate launcher
│       └── fsdp_8gpu.yaml           # 8-GPU FSDP SHARD_GRAD_OP
├── scripts/
│   ├── 00_inspect_dataset.py        # Phase 0b dataset audit
│   ├── 01_smoke_one_step.sh         # 1-GPU smoke test (5 steps, interactive)
│   ├── 02_sbatch_smoke.sh           # SLURM: 1-GPU smoke test (sbatch)
│   ├── 10_train_8gpu.sh             # 8-GPU full SFT launcher (interactive)
│   └── 11_sbatch_train_8gpu.sh      # SLURM: 8-GPU full SFT (sbatch)
├── diffume/
│   ├── tokens.py                    # special-token registration
│   ├── data/
│   │   ├── prompting.py             # template + label/mask builders
│   │   ├── sft_dataset.py           # DiffumePairedDataset
│   │   └── collator.py              # DiffumeCollator
│   ├── diffusion/
│   │   ├── pbnc.py                  # PBNCSampler (SDAR-VL §3.3)
│   │   ├── noising.py               # apply_blockwise_noise (ABNS, §3.1)
│   │   ├── loss.py                  # diffusion_ce_loss_emrs (EMRS, §3.2)
│   │   └── sampling.py              # block_diffusion_generate (inference)
│   ├── models/
│   │   ├── attention_block.py       # build_block_causal_mask
│   │   ├── readout.py               # get_embedding_reps
│   │   └── diffume_model.py         # DiffumeModel (main wrapper)
│   ├── losses/
│   │   └── infonce.py               # ClipLoss (symmetric InfoNCE)
│   ├── training/
│   │   ├── trainer_step.py          # DiffumeStep (per-step loss computation)
│   │   └── train.py                 # Accelerate training entrypoint
│   ├── inference/
│   │   ├── embed_disc.py            # clean-forward discriminative embedding
│   │   └── embed_gen.py             # diffusion-decoded generative embedding
│   └── eval/
│       └── retrieval.py             # Recall@{1,5,10} evaluator
└── tests/
    ├── conftest.py
    ├── test_block_attention.py
    ├── test_pbnc_emrs.py
    ├── test_prompting.py
    └── test_loss_readout.py
```

---

## 3. BD3 background

Block Discrete Denoising Diffusion (BD3) partitions a token sequence of length
`L` into `B` non-overlapping blocks of length `L' = block_size`.  The model
factorises the likelihood autoregressively **over blocks**:

$$\log p_\theta(\mathbf{x}) = \sum_{b=1}^{B} \log p_\theta(x_b \mid x_{<b})$$

Each per-block conditional is modelled by a masked-diffusion process: a
forward process gradually replaces tokens in block `b` with `<|mask|>`, and a
reverse process (the transformer with block-causal attention) reconstructs them.

The standard BD3 training loss (NELBO) is:

$$\mathcal{L}(\theta) = \mathbb{E}_{x,b,t} \left[ -\frac{1}{t} \sum_{\ell \in \mathcal{M}_t^b} \log p_\theta(x_{b,\ell} \mid x_t^b, x_{<b}) \right]$$

SDAR-VL adds three improvements on top of this baseline (see §5.3).

---

## 4. Architecture overview

### 4.1 Full training pipeline (blockwise diffusion)

```
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  INPUT: one (qry, pos) pair per sample                                       ║
 ║                                                                              ║
 ║  Raw conversation  +  PIL image                                              ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
  ┌───────────────────┐           ┌───────────────────┐
  │  Qwen2.5-VL       │           │  Text tokeniser   │
  │  ImageProcessor   │           │  (ChatML template)│
  │                   │           │                   │
  │  pixel_values     │           │  input_ids (1-D)  │
  │  (N_patches, C)   │           │  labels (-100 /   │
  │  image_grid_thw   │           │   token_id)       │
  │  (1, 3)           │           │  assistant_start  │
  └────────┬──────────┘           │  force_mask       │
           │                      │  readout_keep     │
           │                      │  diffusable       │
           └──────────────────────┘
                           │
                           ▼
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  PBNC  (Progressive Beta Noise Curriculum)                  pbnc.py         ║
 ║                                                                              ║
 ║  progress τ ∈ [0,1]  →  Beta(μ_τ·C_τ, (1−μ_τ)·C_τ)                        ║
 ║  sample t_b ~ Beta  for each block b                                        ║
 ║  (μ grows 0.5→0.7, C grows 4→25 over training)                              ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
                           │  t_b per block
                           ▼
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  ABNS  (Asynchronous Block-wise Noise Schedule)             noising.py      ║
 ║                                                                              ║
 ║  assistant span partitioned into blocks of block_size=32 tokens             ║
 ║                                                                              ║
 ║  prefix  [bk=-1] │ block_0 [bk=0] │ block_1 [bk=1] │ … │ block_B           ║
 ║  ─────────────── │ ─────────────── │ ─────────────── │   │ ──────────────── ║
 ║  unchanged        │ Bernoulli(t_0)  │ Bernoulli(t_1)  │   │ Bernoulli(t_B)  ║
 ║                   │ → replace with  │ → replace with  │   │ → replace with  ║
 ║                   │   <|mask|>      │   <|mask|>      │   │   <|mask|>      ║
 ║                                                                              ║
 ║  overrides:  force_mask positions → always <|mask|>                         ║
 ║              readout_keep positions → always verbatim                       ║
 ║                                                                              ║
 ║  output: noisy_ids, rand_mask, block_idx, block_lens, t_realised            ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
                           │  noisy_ids  (B, L)
                           ▼
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  Block-Causal Attention Mask                                attention_block.py║
 ║                                                                              ║
 ║  build_block_causal_mask(block_idx)  →  4-D additive float mask  (B,1,L,L) ║
 ║                                                                              ║
 ║  mask[bq, bk] =  0    if allowed                                            ║
 ║                 -inf  if blocked                                             ║
 ║                                                                              ║
 ║  Rules:                                                                      ║
 ║    prefix→prefix    : standard causal (key_pos ≤ query_pos)                 ║
 ║    assistant→prefix : always visible (full prefix context)                  ║
 ║    assistant→asst   : block-causal  bk ≤ bq  (same block = bidirectional)   ║
 ║    prefix→assistant : never visible                                          ║
 ║                                                                              ║
 ║  Passed as attention_mask to Qwen2_5_VLModel.forward();                     ║
 ║  transformers ≥ 4.45 detects 4-D shape and bypasses its causal-mask builder  ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
                           │  attn_mask_4d  +  noisy_ids  +  position_ids
                           ▼
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  Qwen2.5-VL-3B Backbone  (weights unchanged — no layer modification)        ║
 ║                                                                              ║
 ║  ┌─────────────────────────────────────────────────────────────────────┐    ║
 ║  │  Vision Tower  (backbone.model.visual)                              │    ║
 ║  │  32 × Qwen2VisionBlock  (ViT-style, SiGLIP-like)                   │    ║
 ║  │  patch size 14×14 px, temporal_patch_size=2                        │    ║
 ║  │  pixel_values (N_patches, 1176)  →  patch_embeds (N_patches, 1152) │    ║
 ║  │                         │                                          │    ║
 ║  │             MLP Merger  (merger.mlp + merger.ln_q)                 │    ║
 ║  │             (N_patches, 1152) → (N_pad, 2048)                      │    ║
 ║  │             merge_size=2: 4 patches → 1 image token                │    ║
 ║  └───────────────────────┬─────────────────────────────────────────────┘   ║
 ║                           │  image tokens injected into token stream        ║
 ║  ┌────────────────────────▼────────────────────────────────────────────┐   ║
 ║  │  Language Model  (backbone.model.language_model)                    │   ║
 ║  │  embed_tokens  (vocab=151668+3, hidden=2048)                        │   ║
 ║  │                                                                     │   ║
 ║  │  28 × Qwen2_5_VLDecoderLayer  ← FSDP-wrapped individually          │   ║
 ║  │  ┌──────────────────────────────────────────────────────────────┐  │   ║
 ║  │  │  SdpaAttention  (GQA: 16 heads / 8 kv-heads, head_dim=128)  │  │   ║
 ║  │  │  + 3-D RoPE (position_ids from get_rope_index)              │  │   ║
 ║  │  │  + block-causal 4-D mask  ←──────────────────────────────── │  │   ║
 ║  │  ├──────────────────────────────────────────────────────────────┤  │   ║
 ║  │  │  MLP  (gate_proj / up_proj / down_proj, intermediate=11008) │  │   ║
 ║  │  └──────────────────────────────────────────────────────────────┘  │   ║
 ║  │                         ×28                                         │   ║
 ║  │  norm  →  last_hidden_state  (B, L, 2048)                          │   ║
 ║  │  lm_head  →  logits  (B, L, 151671)                               │   ║
 ║  └─────────────────────────────────────────────────────────────────────┘   ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
              │  last_hidden_state             │  logits
              │  (B, L, 2048)                  │  (B, L, V)
              ▼                                ▼
 ╔══════════════╗               ╔══════════════════════════════════════════════╗
 ║  Readout     ║               ║  EMRS Diffusion Loss                loss.py ║
 ║  readout.py  ║               ║                                              ║
 ║              ║               ║  for each block b:                           ║
 ║  at <disc_emb>:              ║    L_b = mean CE over rand_mask positions    ║
 ║  z_disc (B,2048)             ║    (EMRS: divide by realised t_b', not t_b)  ║
 ║  L2-normalised               ║                                              ║
 ║              ║               ║  L_diff = mean over valid blocks & batch     ║
 ║  at <gen_emb>:               ╚══════════════════════════════════════════════╝
 ║  z_gen  (B,2048)                              │  L_diff_q / L_diff_p
 ║  L2-normalised               ┌────────────────┘
 ╚══════════════╝               │
   │   z_disc_qry, z_disc_pos   │
   │   z_gen_qry,  z_gen_pos    │
   ▼                            ▼
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  Combined Loss                                              trainer_step.py ║
 ║                                                                              ║
 ║  L_disc = ClipLoss(z_disc_qry, z_disc_pos)   symmetric InfoNCE              ║
 ║  L_gen  = ClipLoss(z_gen_qry,  z_gen_pos)    symmetric InfoNCE              ║
 ║                                                                              ║
 ║  L_total = λ_diff·(L_diff_q + L_diff_p) + λ_disc·L_disc + λ_gen·L_gen     ║
 ║          = 1.0  ·  (   ·   +   ·   )   + 1.0  ·   ·     + 1.0  ·   ·      ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
                           │
                           ▼
                   accel.backward(L_total)
                   FSDP all-reduce gradients
                   AdamW step  (lr=1e-5, β=(0.9, 0.95))
                   cosine LR schedule
```

### 4.2 What layers were changed to block-wise diffusion

**No Qwen2.5-VL parameters or layer code was modified.**  The diffusion
behaviour is injected entirely from the outside:

| Intervention | Where | Effect |
|---|---|---|
| **Noisy `input_ids`** | `noising.py` → `backbone.model(input_ids=noisy_ids)` | Assistant tokens are stochastically replaced with `<\|mask\|>` before every forward pass; the model learns to denoise them |
| **Block-causal 4-D attention mask** | `attention_block.py` → passed as `attention_mask` arg | Overrides the transformer's built-in causal mask; tokens within the same block attend bidirectionally, across blocks causally. Applied inside every existing `SdpaAttention` layer — no code change to those layers |
| **3-D RoPE position_ids** | `diffume_model._compute_position_ids` → passed as `position_ids` arg | Computed from clean ids (so image-patch positions are correct even for noised sequences); injected into the existing RoPE computation inside every decoder layer — no code change |
| **Two new vocab rows** (`<gen_emb>`, `<disc_emb>`, `<\|mask\|>`) | `tokens.py` → `model.resize_token_embeddings(151668)` | Appends 3 new rows to `embed_tokens.weight` and `lm_head.weight`; the rest of both matrices are unchanged |
| **EMRS loss** | `loss.py` | Replaces the standard next-token CE; the model's `lm_head` output is re-used but a different subset of positions is supervised |
| **FSDP wrapping** | `train.py` → `accel.prepare(model)` | `Qwen2_5_VLDecoderLayer` instances become child FSDP units; no change to their internal logic |

Summary: the 32 ViT blocks, 28 decoder layers, MLP merger, `embed_tokens`, `lm_head`, and all layer-norm/projection weights are **used as-is** from the pretrained checkpoint.  Only the *attention mask*, *input token ids*, and *position ids* are modified at call time.

### 4.3 Token layout

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>   ← labels=-100
<|im_start|>user\n<|vision_start|>…<|vision_end|>{text}      ← labels=-100
<disc_emb>                                                    ← labels=-100, readout_keep=True
Represent the above input…<|im_end|>                          ← labels=-100
<|im_start|>assistant\n                                       ← labels=-100 (first 3 tok)
<think>{reasoning}</think><answer>{summary}</answer>          ← labels=token_id, diffusable=True
                                                                 <think>/<\/think> ⇒ force_mask=True
<gen_emb>                                                     ← labels=-100, readout_keep=True
<|im_end|>                                                    ← labels=token_id, diffusable=True
```

---

## 5. Module reference

### 5.1 Special tokens — `diffume/tokens.py`

Three special tokens are added to Qwen2.5-VL's vocabulary at runtime:

| Token | ID (assigned at init) | Role |
|---|---|---|
| `<\|mask\|>` | 151665 | Discrete diffusion mask |
| `<gen_emb>` | 151666 | Readout position: generative embedding |
| `<disc_emb>` | 151667 | Readout position: discriminative embedding |

**`register_diffume_special_tokens(model, tokenizer) → dict`**

- Checks whether each token already exists in the tokenizer vocabulary.
- Adds absent tokens via `tokenizer.add_special_tokens(...)`.
- Resizes both the input embedding table (`embed_tokens`) and the output
  projection head (`lm_head`) by calling `model.resize_token_embeddings`.
- Initialises the new rows with the **mean** of all existing embedding rows
  (Brown et al. tokenizer-extension heuristic), applied to both input and
  output embeddings independently (respects un-tied weights correctly).
- Returns `{"mask_id": …, "gen_emb_id": …, "disc_emb_id": …, "n_added": …}`.

---

### 5.2 Data pipeline

#### `diffume/data/prompting.py`

Handles conversation parsing, ChatML string assembly, and the construction of
all per-token boolean masks that control the diffusion and loss.

**`extract_turns(sub) → (user_text, assistant_text)`**

Parses the UME-R1 columnar conversation format `{"from": [...], "value": [...]}` as
well as the more common `[{"role": …, "content": …}]` list-of-dict format.
Raises `ValueError` if either turn is absent.

**`replace_visual_placeholders(user_text, grid_thw_image, grid_thw_video) → str`**

Replaces `<image>` / `<video>` markers in the raw user string with the
Qwen2.5-VL `<|vision_start|><|image_pad|>…<|vision_end|>` blocks.  The number
of `<|image_pad|>` tokens is computed from the image grid dimensions:
`n_pad = (t × h × w) / merge²` where `merge` defaults to 2.

**`tokenize_with_assistant_span(tokenizer, user_text, asst_text, system_message) → dict`**

Assembles the full ChatML string, tokenises it, and records:
- `input_ids` (1-D LongTensor)
- `assistant_start`, `assistant_end` (int scalars, token indices)

**`build_force_masks(input_ids, assistant_start, assistant_end, gen_emb_id, disc_emb_id, think_ids) → dict`**

Given tokenised output, builds four same-length 1-D boolean/long tensors:

| Tensor | True/set at | Meaning |
|---|---|---|
| `labels` | assistant span only; `-100` elsewhere | Loss targets |
| `force_mask` | `<think>` / `</think>` positions (SDAR-VL §4.1 CoT masking) | Always replace with `<\|mask\|>` |
| `readout_keep` | `<gen_emb>` / `<disc_emb>` positions | Never replace; always visible verbatim |
| `diffusable` | assistant span minus `readout_keep` positions | Eligible for stochastic Bernoulli masking |

---

#### `diffume/data/sft_dataset.py` — `DiffumePairedDataset`

Wraps the HuggingFace `umer1_sft_processed` dataset.

**Schema** (one row):
```python
{"dataset_name": str, "qry": Sub, "pos": Sub}
Sub = {"image": PIL.Image | None, "video": list | None,
       "conversations": {"from": list, "value": list}}
```

`__getitem__` calls `_process_side(sub)` for both `qry` and `pos`, returning:
```python
{"qry": {...}, "pos": {...}}
```

Each side dict contains: `input_ids`, `labels`, `force_mask`, `readout_keep`,
`diffusable`, `assistant_start`, `assistant_end`, and optionally
`pixel_values`, `image_grid_thw`.

**Video handling**: the current UME-R1 dump stores 28×28 pixel placeholder
images in the `video` field; these are skipped automatically (`skip_video=True`
by default).  `<video>` markers are stripped from the user text.

---

#### `diffume/data/collator.py` — `DiffumeCollator`

`__call__(batch) → {"qry": side_dict, "pos": side_dict}`

Pads each 1-D tensor in the batch to the maximum length.  Pixel values are
concatenated along the patch axis (`torch.cat(pvs, dim=0)`) following
Qwen2.5-VL convention; the corresponding `image_grid_thw` tensors are
concatenated along dim 0.  The `attention_mask` (2-D padding mask) is derived
from `input_ids != pad_token_id`.

---

### 5.3 Diffusion core

#### `diffume/diffusion/pbnc.py` — Noise schedulers (uniform / Beta-PBNC / clamp)

Implements SDAR-VL's noise scheduler family (`finetuning_args.noise_scheduler_type`).
Selected via `cfg.scheduler_type` ∈ {`uniform`, `beta`, `clamp`}.

**`uniform`** — `t_b ~ U(0, 1)` independently per block.

**`beta`** — *P-rectify Progressive Beta Noise Curriculum*.  The target mean
`μ` is **held fixed** at `target_mean` (default 0.8); the concentration `C`
ramps **linearly** from `start_C` to `end_C` over `warmup_steps` optimiser
steps.  Before the first warmup step the schedule falls back to **uniform**
sampling — the ramp is what makes the curriculum "progressive".

$$C_\tau = C_0 + (C_T - C_0) \cdot \min(\tfrac{\text{step}}{\text{warmup\_steps}}, 1)$$
$$\alpha = \mu \cdot C_\tau, \qquad \beta = (1 - \mu) \cdot C_\tau, \qquad t_b \sim \text{Beta}(\alpha, \beta)$$

**`clamp`** — uniform sample then clamped to `[noise_min, noise_max]`.

Default schedule (matches SDAR-VL stage-1 YAML):

| Hyperparameter | Value |
|---|---|
| `scheduler_type` | `beta` |
| `target_mean` (μ) | 0.8 |
| `start_C` | 2.0 |
| `end_C` | 50.0 |
| `warmup_steps` | 8090 |
| `noise_min` / `noise_max` | 0.0 / 1.0 (clamp only) |

**`PBNCSampler.sample(shape, step, device, dtype) → Tensor`**

Returns independent samples of the requested shape.  Called once per
training step with `shape = (batch_size, max_blocks)` from `apply_blockwise_noise`,
passing the current global optimiser `step` (not progress fraction) so the
C-ramp matches the SDAR-VL reference implementation exactly.

---

#### `diffume/diffusion/noising.py` — ABNS blockwise noising

Implements SDAR-VL §3.1 (Asynchronous Block-wise Noise Schedule).

**`apply_blockwise_noise(...) → dict`**

Takes the clean `input_ids` batch, assistant span boundaries, and the
per-token constraint masks, and returns:

| Key | Shape | Description |
|---|---|---|
| `noisy_ids` | `(B, L)` | `input_ids` with masked positions replaced by `mask_id` |
| `is_masked` | `(B, L)` bool | True where `mask_id` was written (excludes `readout_keep`) |
| `rand_mask` | `(B, L)` bool | Stochastic Bernoulli mask only (excludes `force_mask`) |
| `block_idx` | `(B, L)` long | Block id ∈ [0, B_max), -1 outside assistant span |
| `t_realised` | `(B, B_max)` float | Realised mask ratio per block: `‖m_b‖₁ / L_b` |
| `block_lens` | `(B, B_max)` long | Number of diffusable tokens per block |

**Masking logic** (in priority order):
1. `readout_keep` positions → **never** replace (preserved verbatim).
2. `force_mask` positions (e.g. `<think>`, `</think>`) → **always** replace.
3. `diffusable` positions: Bernoulli(`t_b`) draw → replace if `u < t_b`.
4. All other positions (prefix, padding) → unchanged.

---

#### `diffume/diffusion/loss.py` — EMRS-corrected block diffusion loss

Implements SDAR-VL §3.2, Eq. (6)–(7).

**`diffusion_ce_loss_emrs(logits, targets, rand_mask, block_idx, block_lens) → scalar`**

For each block `b` in each sample:

$$\text{per\_block}_{b} = \frac{\ell_b}{t_b'} = \frac{\sum_{i \in \mathcal{M}_b} \text{CE}_i \;/\; L_b}{t_b'}, \qquad t_b' = \frac{|\mathcal{M}_b|}{L_b}$$

simplifies to:

$$\text{per\_block}_{b} = \frac{\sum_{i \in \mathcal{M}_b} \text{CE}_i}{|\mathcal{M}_b|}$$

i.e. the mean CE over randomly-masked tokens in the block — unweighted by
`t_b'` in the final formula because `ℓ_b / L_b / (|M| / L_b) = ℓ_b / |M|`.

Blocks with zero diffusable tokens or zero masked tokens contribute 0 to the
average.  Final loss = mean over valid blocks, mean over batch.

**Why EMRS?** The standard BD3 scaling `1/t_b` is biased when the intended
ratio `t_b` differs from the realised ratio `t_b' = ‖m_b‖₁/L_b`.  Using `t_b'`
gives an unbiased estimator of the ideal NELBO (paper Lemma 4, Eq. 26).

---

#### `diffume/diffusion/sampling.py` — Block-diffusion inference decoder

**`block_diffusion_generate(model, prefix_input_ids, mask_id, cfg, extra_model_kwargs) → LongTensor`**

Implements SDAR-VL §C.2 low-confidence greedy static decoding:

For each block `b` in order:
1. Initialise block positions to `mask_id`.
2. Run `T` (`cfg.n_steps`) denoising iterations:
   - Full forward pass (prefix + all completed blocks + current all-mask block).
   - Decode only positions in the current block.
   - Keep the `⌈block_size × (1 − t_step)⌉` highest-confidence predictions;
     re-mask the rest (`t_step` decreases linearly each step).
3. Force-decode any remaining `mask_id` positions to argmax.
4. Optionally stop early on `eos_token_id`.

Returns the full sequence `(B, L_pre + n_blocks × block_size)`.

---

### 5.4 Models

#### `diffume/models/attention_block.py` — Block-causal mask

**`build_block_causal_mask(block_idx, dtype, device) → Tensor(B, 1, L, L)`**

Returns an additive float mask with values `{0, -∞}`.  The mask implements
four cases based on the block indices of query `bq` and key `bk`:

| Query | Key | Allowed? |
|---|---|---|
| Prefix (`bq = -1`) | Prefix (`bk = -1`) | Standard causal: `key_pos ≤ query_pos` |
| Assistant (`bq ≥ 0`) | Prefix (`bk = -1`) | Always visible |
| Assistant (`bq ≥ 0`) | Assistant (`bk ≥ 0`) | Block-causal: `bk ≤ bq` (same block = bidirectional) |
| Prefix (`bq = -1`) | Assistant (`bk ≥ 0`) | Never visible |

Passing this as the `attention_mask` argument to `Qwen2_5_VLModel.forward()`
overrides its internal causal mask (Transformers ≥ 4.45 detects 4-D masks
and passes them through unchanged).

---

#### `diffume/models/readout.py`

**`get_embedding_reps(last_hidden_state, input_ids, embedding_token_id, normalize=True) → (B, D)`**

Direct port of UME-R1 `trainer.py:568-580`.  Finds the **last** occurrence
of `embedding_token_id` in each row of `input_ids` via `torch.max`.  Falls
back to the last sequence position if the token is absent (no silent NaN
propagation).  L2-normalises the result by default.

---

#### `diffume/models/diffume_model.py` — `DiffumeModel`

The central `nn.Module`.  Wraps the Qwen2.5-VL backbone and orchestrates
all training-time and inference-time passes.

```python
DiffumeModel(
    backbone,            # Qwen2_5_VLForConditionalGeneration
    mask_id,             # int
    gen_emb_id,          # int
    disc_emb_id,         # int
    cfg,                 # DiffumeConfig(block_size=32, pbnc=PBNCConfig(...))
)
```

**Key attributes:**
- `self.backbone` — `Qwen2_5_VLForConditionalGeneration`
- `self.sampler` — `PBNCSampler` (per-step PBNC sampling)
- `self.schedule` — `NoiseSchedule(block_size=32)`

**`forward(**kwargs) → dict`** *(FSDP entry point)*

Thin wrapper that calls `forward_train(**kwargs)`.  Must be used as `model(...)`
(not `model.forward_train(...)`) so FSDP's pre-forward hook can gather sharded
parameters before computation.

**`forward_train(...) → dict`**

Full training forward:

1. **Apply block-wise noise** — `apply_blockwise_noise(...)` produces
   `noisy_ids`, `rand_mask`, `block_idx`, `block_lens`, `t_realised`.
2. **Compute 3D RoPE position_ids** — `_compute_position_ids(input_ids, vision_kwargs, attention_mask_2d)`.
   Called with the *clean* `input_ids` (before noising) and the 2-D padding mask.
   Uses `self.backbone.get_rope_index(...)` to assign correct `(t, h, w)`
   positions to image-patch tokens.  Returns `None` for text-only batches.
   This must happen before building the 4-D block-causal mask because
   Qwen2.5-VL's `get_rope_index` only accepts 2-D attention masks.
3. **Build 4-D block-causal mask** — `build_block_causal_mask(block_idx, ...)`,
   then apply padding mask (`-∞` at padding key positions).
4. **Backbone forward** — calls `self.backbone.model(input_ids=noisy_ids,
   attention_mask=attn_mask_4d, position_ids=position_ids, ...)` to get
   `last_hidden_state`.  Then `self.backbone.lm_head(last_hidden_state)` → `logits`.
5. **Returns** the full output dict needed by `DiffumeStep`.

Returns:
```python
{
    "logits":            (B, L, V),
    "last_hidden_state": (B, L, D),
    "noisy_ids":         (B, L),
    "rand_mask":         (B, L) bool,
    "block_idx":         (B, L) long,
    "block_lens":        (B, B_max) long,
    "t_realised":        (B, B_max) float,
}
```

**`get_embeddings(last_hidden_state, input_ids) → (z_disc, z_gen)`**

Reads out the hidden state at `<disc_emb>` and `<gen_emb>` positions.
Called after `forward_train` using the already-computed `last_hidden_state` —
does not access model parameters, so it is safe to call outside the FSDP
forward context.

**`encode(...) → (z_disc, z_gen)`**

Clean (no-noise) inference forward.  Builds a block-causal mask with
`t_b = 0` everywhere (no masking applied), runs the backbone, and returns
the normalised embeddings.  Decorated `@torch.no_grad()`.

---

### 5.5 Losses

#### `diffume/losses/infonce.py` — `ClipLoss`

Symmetric InfoNCE loss, adapted from UME-R1 `trainer.py:449-564`.

```python
ClipLoss(local_loss=True, gather_with_grad=True)
```

**`forward(q, t, logit_scale=50.0) → scalar`**

In a distributed setting, performs `torch.distributed.nn.all_gather` (which
preserves gradients) to collect embeddings from all ranks before computing
the cross-entropy in both directions:

$$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2} \left( \text{CE}(S_{qt}, \mathbf{y}) + \text{CE}(S_{tq}, \mathbf{y}) \right), \qquad S_{qt} = s \cdot Q \cdot T^\top$$

With `local_loss=True`, each rank computes cross-entropy over its local rows
`q` vs all-gathered `all_t`, which matches UME-R1's training semantics.
Falls back to local `q @ t.T` when not in a distributed context.

---

### 5.6 Training

#### `diffume/training/trainer_step.py` — `DiffumeStep`

Owns the per-step loss computation.  Called once per micro-batch inside
`accelerator.accumulate(model)`.

```python
DiffumeStep(model, weights=LossWeights(diff=1.0, disc=1.0, gen=1.0, logit_scale=50.0))
```

**`__call__(batch, progress) → dict`**

1. Calls `_side(batch["qry"], progress)` and `_side(batch["pos"], progress)`.
2. Combines: `loss = λ_diff·(L_diff_q + L_diff_p) + λ_disc·L_disc + λ_gen·L_gen`.
3. Returns 5-key dict: `{loss, loss_diff_q, loss_diff_p, loss_disc, loss_gen}`.

**`_side(side, progress)`**

1. Calls `self.model(...)` — goes through FSDP forward hook.
2. Computes `diffusion_ce_loss_emrs(...)`.
3. Calls `self.model.get_embeddings(last_hidden_state, input_ids)`.
4. Returns `(l_diff, z_disc, z_gen)`.

---

#### `diffume/training/train.py` — Accelerate entrypoint

**`main()`**

1. Parses `--config path/to/yaml`.
2. Builds `Accelerator` with `mixed_precision=bf16`, `gradient_accumulation_steps`.
3. Loads backbone via `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)` with `attn_implementation="sdpa"`.
4. Registers special tokens and wraps in `DiffumeModel`.
5. Loads `umer1_sft_processed` via `datasets.load_from_disk(...)`.
6. Builds `DiffumePairedDataset` → `DiffumeCollator` → `DataLoader`.
7. Builds `AdamW(lr=1e-5, betas=(0.9, 0.95))`.
8. Builds cosine-with-warmup LR schedule via `LambdaLR`.
9. `accel.prepare(model, optim, loader, sched)` — wraps model with FSDP.
10. Constructs `DiffumeStep(model.module, ...)` — note `.module` to unwrap FSDP.
11. Training loop: `accel.accumulate(model)` context → `step_fn(batch, progress)` → `accel.backward(loss)` → grad clip → `optim.step` → `sched.step`.
12. Logs JSON every `log_every` steps; saves `accel.save_state(...)` every `save_every` steps.

**Throughput log format** (one JSON line per `log_every` steps):
```json
{"loss": 1.23, "loss_diff_q": 0.61, "loss_diff_p": 0.62,
 "loss_disc": 0.0, "loss_gen": 0.0,
 "step": 0, "lr": 2e-6, "throughput_steps_per_sec": 0.30}
```

---

### 5.7 Inference

#### `diffume/inference/embed_disc.py`

**`load_diffume(checkpoint_dir, backbone_name, dtype) → (model, tokenizer, processor, ids)`**

Loads backbone, registers tokens, wraps in `DiffumeModel`, and optionally
loads weights from `checkpoint_dir/pytorch_model.bin`.

**`embed_pair(model, tokenizer, processor, ids, *, user_text, assistant_text, image, device) → (z_disc, z_gen)`**

Single-example clean forward:
1. Process image → `pixel_values`, `image_grid_thw`, `n_pad`.
2. Build full ChatML sequence (user + assistant text supplied verbatim).
3. Tokenise → build masks via `build_force_masks`.
4. Call `model.encode(...)` → `(z_disc, z_gen)`.

---

#### `diffume/inference/embed_gen.py`

**`embed_generate(model, tokenizer, processor, ids, *, user_text, image, n_blocks, n_steps, device) → (z_disc, z_gen, full_ids)`**

Generative path: the assistant span is **not** supplied; it is decoded from
scratch by `block_diffusion_generate(...)`.  After decoding:
1. Runs one clean forward pass on the fully decoded sequence.
2. Reads out `<gen_emb>` and `<disc_emb>` hidden states.
3. Returns both embeddings plus the decoded token ids.

---

### 5.8 Evaluation

#### `diffume/eval/retrieval.py`

**`main()`** (CLI: `--checkpoint`, `--dataset`, `--n`, `--backbone`)

Loads `n` pairs from the dataset, embeds each query and positive with
`embed_pair` (discriminative / clean forward), then computes:

$$\text{Recall@}k = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\left[\text{rank of positive}_i \leq k\right]$$

using cosine similarity matrix `Q @ P.T`.  Prints a table for `k ∈ {1, 5, 10}`.

---

## 6. Training details

### Effective batch size

| Setting | Value |
|---|---|
| `per_device_batch_size` | 1 |
| `grad_accum` | 8 |
| GPUs | 8 |
| Effective batch (pairs) | 1 × 8 × 8 = **64 pairs** |

### Optimiser & schedule

- `AdamW(lr=1e-5, β₁=0.9, β₂=0.95, wd=0.0)`
- Cosine decay with `warmup_steps=500`, `min_lr=5%` of peak.
- Gradient clip: `max_norm=1.0`.
- `gradient_checkpointing=True` on the backbone.

### FSDP configuration (`configs/accelerate/fsdp_8gpu.yaml`)

| Setting | Value |
|---|---|
| `fsdp_sharding_strategy` | `FULL_SHARD` (ZeRO-3 equivalent) |
| `fsdp_auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` |
| `fsdp_transformer_layer_cls_to_wrap` | `Qwen2_5_VLDecoderLayer` |
| `fsdp_use_orig_params` | `true` |
| `fsdp_state_dict_type` | `FULL_STATE_DICT` |
| `mixed_precision` | `bf16` |

`FULL_SHARD` shards parameters, gradients, and optimizer states across 8 GPUs.
Each GPU holds only 1/8 of the parameters at rest; parameters are all-gathered
on demand during the forward and backward passes and immediately discarded after
each layer — reducing per-GPU resident parameter memory from ~6 GB to ~0.75 GB.

### Memory budget

| Component | bf16 |
|---|---|
| Qwen2.5-VL-3B weights shard (1/8) | ~0.75 GB |
| Gradient shard (1/8) | ~0.75 GB |
| Adam states shard (1/8) | ~1.5 GB |
| Activations (batch=1, seq=1536) | ~15 GB |
| Gathered layer params during forward | ~1 GB peak |
| **Total per GPU** | **~19 GB** |

Fits on A100-80 GB with `gradient_checkpointing=True` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 7. Configs & scripts

### `configs/diffume_debug_1gpu.yaml` — smoke test

```yaml
max_steps: 5, max_train_samples: 64, per_device_batch_size: 1,
block_size: 32, max_length: 1024, num_workers: 0
```

### `configs/diffume_stage1.yaml` — full SFT

```yaml
max_steps: 30000, per_device_batch_size: 1, grad_accum: 8,
max_length: 1536, num_workers: 4, warmup_steps: 500, save_every: 2000
output_dir: /mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/diffume-stage1
```

### `scripts/01_smoke_one_step.sh` / `scripts/02_sbatch_smoke.sh`

Runs 5 training steps on 1 GPU with the debug config.  Pass criteria:
all loss keys finite for each step; no NaN/Inf.
Logs written to `/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/logs/`.

### `scripts/10_train_8gpu.sh` / `scripts/11_sbatch_train_8gpu.sh`

Full SFT launcher using `configs/accelerate/fsdp_8gpu.yaml` and
`configs/diffume_stage1.yaml`.  Checkpoints saved to
`/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/diffume-stage1/`.

### `scripts/00_inspect_dataset.py`

Dataset audit: prints schema, counts by `dataset_name`, checks that every
assistant turn contains `<gen_emb>`.

---

## 8. Tests

All tests run on CPU (`pytest tests/`).

| File | What it tests |
|---|---|
| `test_block_attention.py` | `build_block_causal_mask`: prefix-only = causal; within-block = bidirectional; across blocks = causal; prefix invisible to assistant queries |
| `test_pbnc_emrs.py` | PBNC α/β linear schedule; samples in [0,1]; ABNS constraints (`force_mask`, `readout_keep`, prefix unchanged); EMRS loss = 0 when no masks; finite + differentiable with masks |
| `test_prompting.py` | Label mask construction; special positions are -100; assistant span has ≥1 non-(-100) target |
| `test_loss_readout.py` | `get_embedding_reps` finds correct position; `diffusion_ce_loss_emrs` gradient flows |

---

## 9. Known deviations from the plan

| Plan item | Actual implementation | Notes |
|---|---|---|
| Vendor `modeling_qwen2_5_vl.py` (plan §3.2) | Pass 4-D additive mask externally; no vendored copy | Simpler; requires transformers ≥ 4.45 |
| `proj_disc` / `proj_gen` projection heads (plan §4b) | No projection heads; `get_embedding_reps` reads directly from `last_hidden_state` | Plan projected to 1024-D; current impl uses raw 2048-D hidden state |
| `forward_single` (plan §4b) | `forward_train` | Renamed; same logic |
| DeepSpeed ZeRO-2 (plan §7) | FSDP `FULL_SHARD` | DeepSpeed not available on GPU nodes at job time; FSDP is functionally equivalent |
| `disc_batch_acc`, `gen_batch_acc` keys (plan §5c) | Not present in current `DiffumeStep` | Batch accuracy monitoring not yet implemented |
| `t_realised` in `noising.py` output (plan §3b) | Computed but not currently threaded through to the loss (loss recomputes it from `rand_mask`) | Redundant recomputation; harmless |
| `<think>`/`</think>` as new special tokens (plan Q2) | Treated as ordinary token sequences | Easier; CoT masking works by token id lookup at mask construction time |

---

## 10. Glossary

| Term | Definition |
|---|---|
| **BD3** | Block Discrete Denoising Diffusion — partitions sequence into blocks; AR across blocks, masked diffusion within blocks |
| **ABNS** | Asynchronous Block-wise Noise Schedule — each block draws its own `t_b` independently (SDAR-VL §3.1) |
| **EMRS** | Effective Mask Ratio Scaling — divide block loss by realised `t_b' = ‖m_b‖₁/L_b` instead of intended `t_b` (SDAR-VL §3.2) |
| **PBNC** | Progressive Beta Noise Curriculum — Beta distribution mean and concentration grow with training progress (SDAR-VL §3.3) |
| **block-causal mask** | Attention mask where tokens attend bidirectionally within a block and causally across blocks |
| **diffusable** | Boolean per-token flag: True iff the token is in the assistant span, is not a readout token, and is not a `<MASK>` placeholder |
| **force_mask** | Boolean flag: True at `<think>` / `</think>` positions; these are always replaced by `<\|mask\|>` regardless of the Bernoulli draw |
| **readout_keep** | Boolean flag: True at `<gen_emb>` / `<disc_emb>` positions; never replaced, hidden state is the embedding output |
| **z_disc** | L2-normalised hidden state at the `<disc_emb>` position; trained via symmetric InfoNCE |
| **z_gen** | L2-normalised hidden state at the `<gen_emb>` position; trained via symmetric InfoNCE |
| **3D RoPE** | Qwen2.5-VL's multimodal positional encoding assigning `(t, h, w)` coordinates to image patches and 1-D positions to text |
| **FSDP** | Fully Sharded Data Parallel; `FULL_SHARD` = ZeRO-3 (shards parameters + gradients + optimizer states) |
