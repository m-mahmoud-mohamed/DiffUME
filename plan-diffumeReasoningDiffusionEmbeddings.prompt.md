# DiffUME Plan — Reasoning-Driven Block-wise Diffusion Multimodal Embeddings

> Replaces all prior versions. **No SDAR-VL repo or weights exist yet.** We
> apply the SDAR-VL recipe (BD3 + ABNS + EMRS + PBNC + CoT template masking)
> directly onto UME-R1's Qwen2.5-VL stack — converting it from autoregressive
> embeddings to **block-wise discrete diffusion embeddings**. Project is
> **standalone**: required files are copied from `UME-R1/` into `DiffUME/`,
> not imported via sys.path.

---

## 0. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Base code | UME-R1 (`src/sft-train`) copied into `DiffUME/diffume/` | Standalone repo, no symlinks |
| Base weights | `Qwen/Qwen2.5-VL-3B-Instruct` (or `-7B-Instruct`) | UME-R1 default; LLaMA-style decoder we can convert to block-diffusion |
| Method | SDAR-VL recipe applied to Qwen2.5-VL | SDAR-VL repo/weights not released; we re-implement |
| Embedding tokens | `<gen_emb>` (assistant suffix) + `<disc_emb>` (user span) | UME-R1 dual-mode parity (`GEN_EMB_ID=151657`, `DISC_EMB_ID=151658`) |
| Loss | `λ_r·(L_diff_q + L_diff_p) + λ_d·ClipLoss(z_disc) + λ_g·ClipLoss(z_gen)` | UME-R1 contrastive + SDAR-VL block-diffusion CE |
| Dataset | `load_from_disk("/mnt/ceph-hdd/cold/nii00224/UME_R1/umer1_sft_processed")` | 1,463,360 paired rows; columns `dataset_name, qry, pos`; reasoning pre-filled |
| Compute | 1×GPU debug → 8×A100 SFT (DeepSpeed ZeRO-2, bf16) | Same as UME-R1 SFT |
| Training | Direct paired SFT only (no separate alignment / CoT / RL stages in v0) | Reasoning is already in the dataset |

---

## 1. Objective

Produce the first runnable, paired-SFT, **block-wise discrete diffusion**
multimodal embedding model — Qwen2.5-VL converted to BD3 with the SDAR-VL
training tricks, trained on the UME-R1 SFT corpus, on 8×A100. Output:
two L2-normalized embedding heads (`disc`, `gen`), evaluated with Recall@k
on a held-out subset.

---

## 2. SDAR-VL plan (applied to Qwen2.5-VL because the SDAR-VL repo does not exist)

This section is the **SDAR-VL re-implementation plan**, derived from
arXiv:2512.14068v1 (Cheng et al., 2025). All notation matches the paper.

### 2.1 Architectural conversion: AR Qwen2.5-VL → BD3

**What changes (vs. stock Qwen2.5-VL):**

1. **Vocabulary**: add a single `<MASK>` token (the diffusion mask), plus
   the embedding tokens `<gen_emb>`, `<disc_emb>`. New rows initialised
   from the mean of existing rows in input embeds + lm_head.
2. **Attention mask**: replace strict causal mask with **block-causal**
   attention:
   - Sequence is partitioned into `B` blocks of length `L' = L/B` (we use
     `L' = 32` for ≤2k seqs; `L' = 64` for packed 16k seqs — paper default).
   - Tokens in block `b` attend bidirectionally to all tokens in block `b`
     and causally to all tokens in blocks `< b`.
   - Tokens in block `b` cannot see any token in block `b+1` or later
     (preserves inter-block AR factorisation).
   - Image / video patch tokens live in their natural position; they belong
     to whichever block their position falls in, and their attention mask
     follows the same block-causal rule.
3. **Forward objective**: replace next-token CE with **masked
   reconstruction CE** on noised positions (per block). Paper Eq. (5) +
   Eq. (7).
4. **Sampler / inference**: replace AR `generate` with block-wise
   denoising — per block, run `T` denoising steps with low-confidence
   greedy static decoding (paper §C.2: block_length=4, T=4 default;
   we use block_length=32, T=8 for embeddings).

**What we keep from Qwen2.5-VL:**

- Vision tower (`Qwen2_5_VLVisionTransformer`) and image processor
  (`process_image_unified` from UME-R1 `data_qwen.py`).
- Tokenizer (`Qwen2Tokenizer`) + chat template (`<|im_start|>` / `<|im_end|>`).
- Decoder layer body (RoPE, SwiGLU MLP, RMSNorm, attention).
- Hidden size, num_layers, num_heads.

### 2.2 The three SDAR-VL training contributions

These are the **only** training-time changes on top of vanilla BD3.
Re-implemented from the paper, file-by-file.

#### (A) Asynchronous Block-wise Noise Schedule — ABNS  *(paper §3.1)*

- For each training sample of `B` blocks, sample **independent**
  per-block noise levels `t_b ∼ P(·|τ)` (Eq. 3).
- For each block, sample binary mask `m_b ∈ {0,1}^{L'}` via independent
  Bernoulli(`t_b`) per position; replace masked positions with `<MASK>`
  token id.
- Asynchronous loss: `L_async = E[ -ℓ_b / t_b ]` (Eq. 5). Replaces
  synchronous BD3 baseline where all blocks share one `t`.
- **Implementation**: in `forward_add_noise_blockwise`, sample
  `t = torch.distributions.Beta(α, β).sample((B,))` then independent
  Bernoulli per block; build noised `input_ids` and a `mask_indicator`
  tensor of shape `(L,)`.
- **Effect** (paper Lemma 2): variance of mini-batch loss reduced by
  `(1 − 1/B)·Var_t(μ(t))`; expectation unchanged (Lemma 1).

#### (B) Effective Mask Ratio Scaling — EMRS  *(paper §3.2)*

- Compute the **realised** mask ratio per block:
  `t_b' = ||m_b||_1 / L'` (Eq. 6).
- Normalize each block's CE by `1/t_b'` instead of `1/t_b` (Eq. 7).
- **Implementation**: 1-line change in the diff-CE wrapper —
  `scale_b = 1.0 / max(t_b_prime, 1.0/L')`. The clamp protects against
  empty masks.
- **Effect** (paper Lemma 4): `1/t_b'` scaling is unbiased for the ideal
  NELBO `L*(θ) = E[ -ℓ_b / t_b' ]` (Eq. 26). Removes the systematic mis-
  scaling of standard BD3 (paper Lemma 3, Eq. 28).

#### (C) Progressive Beta Noise Curriculum — PBNC  *(paper §3.3)*

- Sample noise ratios from a Beta distribution whose **mean `μ_τ`** and
  **concentration `C_τ`** both grow with normalized training progress
  `u_τ ∈ [0,1]`:
  - `α_τ = μ_τ · C_τ`,  `β_τ = (1 - μ_τ) · C_τ`.
  - `μ_τ` linear from `μ_0 = 0.5` → `μ_final = 0.7`.
  - `C_τ` linear from `C_0 = 4` → `C_final = 25` (reasoning/text-rich
    regime per paper Table 4) **or** `C_final = 50` (general-understanding
    regime). Default v0: **C_final = 25**.
- **Implementation**: `PBNCSampler` class with `update(global_step,
  total_steps)` → recomputes `(α, β)` and exposes `.sample(B)`. Lives
  in `diffume/diffusion/pbnc.py`.
- **Effect** (paper §4.4.3): best Mat heVista / ChartQA / DocVQA scores
  vs. SNS, ABNS, ABNS+EMRS, and Clamp baselines.

### 2.3 CoT template masking  *(paper §4.1, "Chain-of-Thought Template Masking")*

- Tokens `<think>` and `</think>` are **always** in the mask set during
  training, regardless of the per-block Bernoulli draw. Provides direct
  supervision over reasoning-boundary placement.
- We extend this to the embedding setting: `<gen_emb>` and `<disc_emb>`
  are **always** in the `labels = -100` set (never noised, never
  contribute to diff-CE) — they are pure read-out positions.

### 2.4 Block-wise inference (paper §C.2)

- **For training-time embedding readout**: a single forward pass with
  no noise (all `t_b = 0`) on the full clean sequence, gather hidden
  state at `<disc_emb>` / `<gen_emb>`. No denoising loop required at
  train step.
- **For inference-time gen embedding** (eval / deployment): block-wise
  low-confidence greedy static decoding — for each block in order:
  1. Initialize block to all `<MASK>`.
  2. Run `T` denoising steps; at each step keep top-`k` highest-confidence
     positions, re-mask the rest.
  3. Move to next block (block-causal attention sees previously decoded
     blocks as clean).
  Default `T = 8`, block_length = 32 for embeddings.
- Implemented in `diffume/diffusion/sampling.py:block_diffusion_generate`.

---

## 3. Standalone project structure

```
DiffUME/
├── pyproject.toml                # all deps pinned; standalone install
├── README.md
├── plan-diffumeReasoningDiffusionEmbeddings.prompt.md   ← this file
├── docs/
│   ├── design.md                 # frozen design doc (mirror of §2)
│   └── dataset_report.md         # generated by scripts/00_inspect_dataset.py
├── diffume/
│   ├── __init__.py
│   │
│   ├── tokens.py                 # MASK_TOKEN, GEN_EMB_TOKEN, DISC_EMB_TOKEN; add_special_tokens
│   │
│   ├── data/                     # COPIED & ADAPTED from UME-R1/src/sft-train/qwenvl/data/
│   │   ├── data_qwen.py          # process_image_unified, image grid logic (verbatim from UME-R1)
│   │   ├── prompting.py          # build_conversation, build_labels_mask (NEW; UME-R1 template)
│   │   ├── sft_dataset.py        # DiffUMESFTDataset (paired qry/pos from load_from_disk)
│   │   └── collator.py           # DiffUMECollator → (2B, L) batch
│   │
│   ├── models/                   # COPIED & MODIFIED from transformers Qwen2.5-VL
│   │   ├── qwen2_5_vl_block_diff.py   # Qwen2_5_VL with block-causal mask + masked-CE forward
│   │   ├── attention_block.py    # block-causal mask builder
│   │   ├── readout.py            # get_embedding_reps (port of UME-R1 trainer.py:568-580)
│   │   └── diffume_model.py      # DiffUMEModel: backbone + proj_disc + proj_gen
│   │
│   ├── diffusion/                # SDAR-VL re-implementation (NEW)
│   │   ├── pbnc.py               # PBNCSampler (Progressive Beta Noise Curriculum)
│   │   ├── noising.py            # forward_add_noise_blockwise (ABNS sampler + masking)
│   │   ├── loss.py               # diffusion_ce_loss (EMRS unbiased scaling)
│   │   └── sampling.py           # block_diffusion_generate (inference-time T-step denoiser)
│   │
│   ├── losses/
│   │   └── infonce.py            # ClipLoss (re-derived from UME-R1 trainer.py:449-564)
│   │
│   ├── training/
│   │   ├── trainer_step.py       # compute_loss → 7-key dict
│   │   └── train.py              # Accelerate entrypoint
│   │
│   ├── inference/
│   │   ├── embed_disc.py         # one-pass clean forward → readout at <disc_emb>
│   │   └── embed_gen.py          # T-step block denoising → readout at <gen_emb>
│   │
│   └── eval/
│       └── retrieval.py          # Recall@{1,5,10} per dataset_name
│
├── configs/
│   ├── diffume_v0_debug_1gpu.yaml
│   ├── diffume_v0_8gpu.yaml
│   └── accelerate/
│       ├── 1_gpu.yaml
│       └── 8_gpu_zero2.yaml
├── scripts/
│   ├── 00_inspect_dataset.py
│   ├── 01_smoke_one_step.sh
│   ├── 10_train_8gpu.sh
│   └── 20_eval_retrieval.sh
└── tests/
    ├── test_tokens.py
    ├── test_prompting.py
    ├── test_collator.py
    ├── test_dataset.py
    ├── test_block_attention.py    # block-causal mask unit test
    ├── test_pbnc.py               # Beta sampler curriculum
    ├── test_noising.py            # ABNS mask-ratio statistics
    ├── test_emrs_unbiased.py      # check 1/t_b' scaling matches paper Lemma 4
    ├── test_readout.py
    ├── test_losses_shapes.py
    └── test_one_step_cpu.py       # tiny full forward+backward integration
```

### 3.1 Files copied from UME-R1 (verbatim or near-verbatim)

| Source (UME-R1)                                            | Destination (DiffUME)                       | Modification |
|---|---|---|
| `src/sft-train/qwenvl/data/data_qwen.py`                   | `diffume/data/data_qwen.py`                 | Keep `process_image_unified`, `LazySupervisedDataset` retry pattern; drop AR-only assumptions |
| `src/sft-train/qwenvl/train/trainer.py` lines 449-564      | `diffume/losses/infonce.py` (re-derived)    | Standalone re-implementation of `ClipLoss(local_loss=True, gather_with_grad=True, logit_scale=50)` |
| `src/sft-train/qwenvl/train/trainer.py` lines 568-580      | `diffume/models/readout.py`                 | Verbatim port of `get_embedding_reps`; raise on token absent |
| `src/sft-train/qwenvl/train/trainer.py` lines 583-630      | `diffume/training/trainer_step.py`          | Adapted: paired `forward(qry, pos)` returning 7-key loss dict |
| `src/r1-train/src/open_r1/vlm_modules/qwen_module.py:75`   | `diffume/data/prompting.py`                 | Embed instruction string verbatim |
| `src/sft-train/qwenvl/data/data_qwen.py:preprocess_qwen_2_visual` | `diffume/data/prompting.py:build_labels_mask` | Adapted to mask special-embed-token positions to -100 |

### 3.2 Files copied from HuggingFace transformers (forked + modified)

| Source                                                  | Destination (DiffUME)                              | Modification |
|---|---|---|
| `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` | `diffume/models/qwen2_5_vl_block_diff.py`          | Replace causal mask → block-causal; replace next-token CE → masked-recon CE driven by `mask_indicator` |
| (within same file) attention mask builder               | `diffume/models/attention_block.py`                | New `build_block_causal_mask(L, block_size, attn_pad)` |

---

## 4. Template (UME-R1 dual-mode, unchanged from prior plan — verified against `qwen_module.py:75` + `preprocess_qwen_2_visual`)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>…<|vision_end|>{question_text} 
<disc_emb>
Represent the above input text, images, videos, or any combination of the
three as embeddings. First output the thinking process in <think> </think>
tags and then summarize the entire input in a word or sentence. Finally,
use the <gen_emb> tag to represent the entire input.<|im_end|>
<|im_start|>assistant
<think>{reasoning}</think><answer>{summary}</answer><gen_emb><|im_end|>
```

- System + user spans: `labels = -100` everywhere; `<disc_emb>` likewise -100
  but its position is recorded for the discriminative readout.
- Assistant span: first 3 tokens (`<|im_start|>assistant\n`) = -100; then
  `<think>…</think><answer>…</answer>` = token-id labels (these are the
  diffusion targets and the only positions that get noised); `<gen_emb>` = -100
  (read-out only). `<|im_end|>` = token-id label (diff target).
- **Block partitioning**: blocks are created **after** template + image
  expansion, on the final flat token sequence. Block boundaries are not
  aligned to template structure — they are uniform stride `L'`. The
  block-causal mask still respects the template because the diffusion CE
  is gated on `labels != -100`.

---

## 5. Loss

```
L = λ_r·(L_diff_q + L_diff_p)        # block-diffusion CE with EMRS, per side
  + λ_d·ClipLoss(z_disc_q, z_disc_p) # in-batch InfoNCE on <disc_emb> readout
  + λ_g·ClipLoss(z_gen_q,  z_gen_p)  # in-batch InfoNCE on <gen_emb> readout
```

- `λ_r = λ_d = λ_g = 1.0` v0 default. Sweep deferred.
- `ClipLoss`: symmetric InfoNCE, cross-rank `all_gather` with grad,
  `logit_scale = 50` (UME-R1 parity).
- `L_diff_*`: paper Eq. (7) — sum over blocks of `(1/t_b') · Σ_{ℓ ∈ M_b} CE(x_b^ℓ)`
  where `M_b` is the realised masked position set for block `b`, **intersected
  with `labels != -100`** (so user/system/special tokens never contribute).

---

## 6. Step-by-step implementation roadmap

### Phase 0 — Standalone bootstrap & dataset audit  *(blocking gate)*

**0a. Repo skeleton + copy UME-R1 files**
- `mkdir -p DiffUME/{diffume/{data,models,diffusion,losses,training,inference,eval},configs/accelerate,scripts,tests,docs,outputs}`
- Copy listed UME-R1 files into `diffume/` per §3.1 (no symlinks).
- Vendor `transformers` Qwen2.5-VL modeling file into `diffume/models/qwen2_5_vl_block_diff.py` for surgery (or subclass at runtime — decide in Phase 3).
- `pyproject.toml`: pin `torch>=2.1`, `transformers==4.46.0`, `accelerate`, `deepspeed`, `datasets`, `wandb`, `triton`, `pillow`, `qwen_vl_utils`.

**0b. Dataset audit — `scripts/00_inspect_dataset.py`**
- Confirm `qry` / `pos` sub-dict schema (`conversations`, `image`/`video`, `data_path`).
- Confirm assistant turn already contains `<think>…</think><answer>…</answer><gen_emb>` (reasoning pre-filled).
- `dataset_name` distribution; image-only vs video-only vs text-only counts.
- Output → `docs/dataset_report.md`. **Gate before Phase 1.**

**0c. Backbone load check (CPU)**
- Load `Qwen/Qwen2.5-VL-3B-Instruct` via `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`.
- Verify image-processor + tokenizer round-trip with the UME-R1 template.

---

### Phase 1 — Tokens & prompting  *(parallel; depends on 0)*

**1a. `diffume/tokens.py`**
- `MASK_TOKEN = "<MASK>"`, `GEN_EMB_TOKEN = "<gen_emb>"`, `DISC_EMB_TOKEN = "<disc_emb>"`.
- `add_special_tokens(tokenizer, model) → (mask_id, gen_emb_id, disc_emb_id)`. Resize input embeds + lm_head; init new rows = mean of existing rows.
- **Test** `tests/test_tokens.py`: 3 new ids ∉ original vocab; embed shape = `(orig+3, H)`; rows non-zero.

**1b. `diffume/data/prompting.py`**
- `EMBED_INSTRUCTION = "Represent the above input text..."` (verbatim from `qwen_module.py:75`).
- `build_conversation(question_text, thinking, summary)`.
- `build_labels_mask(input_ids, tokenizer, gen_emb_id, disc_emb_id) → labels`: system/user = -100, first 3 assistant tokens = -100, `<gen_emb>`/`<disc_emb>` = -100, rest = token id. Raises `ValueError` if either special token absent.
- **Test** `tests/test_prompting.py`: special positions = -100; ≥1 assistant token ≠ -100; all user tokens = -100.

---

### Phase 2 — Block-causal attention + Qwen2.5-VL surgery  *(blocking; depends on 0c)*

**2a. `diffume/models/attention_block.py`**
- `build_block_causal_mask(seq_len, block_size, attention_pad) → BoolTensor (1, 1, L, L)`:
  - `block_id[i] = i // block_size`.
  - `mask[i,j] = (block_id[j] < block_id[i]) OR (block_id[j] == block_id[i])` — i.e., bidirectional within block, causal across blocks.
  - Combine with attention padding mask.
- **Test** `tests/test_block_attention.py`: `L=8, block_size=4` → assert mask matrix matches the paper's block-causal pattern; assert AR causal recovered when `block_size=1`.

**2b. `diffume/models/qwen2_5_vl_block_diff.py`**
- Subclass / fork `Qwen2_5_VLForConditionalGeneration`:
  - Override `_prepare_4d_causal_attention_mask` to call `build_block_causal_mask`.
  - Add `forward_add_noise_blockwise(input_ids, labels, pixel_values=..., t_per_block, mask_indicator)` returning `(diff_ce_loss, last_hidden_state)`.
  - **Diffusion CE** computed inside via the EMRS rule (calls `diffume/diffusion/loss.py`).
  - Capture final hidden state via a one-shot forward hook on the last decoder layer (avoids `output_hidden_states=True` memory doubling).
- **Test**: `tests/test_one_step_cpu.py` (later, in Phase 5) — full forward+backward.

---

### Phase 3 — Diffusion ABNS + EMRS + PBNC  *(parallel 3a ‖ 3b ‖ 3c; depends on 1a)*

**3a. `diffume/diffusion/pbnc.py`**
- `class PBNCSampler`: stores `(μ_0, μ_final, C_0, C_final, total_steps)`; `update(step)` recomputes `α, β`; `sample(B) → t (B,)` from `Beta(α, β)`.
- Defaults: `μ_0=0.5, μ_final=0.7, C_0=4, C_final=25` (paper Table 4 reasoning regime).
- **Test** `tests/test_pbnc.py`: at step 0, `mean(t) ≈ 0.5`, broad spread; at step `total`, `mean(t) ≈ 0.7`, narrower spread; both monotone in step.

**3b. `diffume/diffusion/noising.py`**
- `apply_blockwise_noise(input_ids, labels, mask_token_id, block_size, t_per_block, always_mask_token_ids=()) → (noised_input_ids, mask_indicator, t_prime_per_block)`:
  - For each block `b`: Bernoulli(`t_b`) draw → `m_b ∈ {0,1}^{L'}`.
  - **Force-mask**: positions whose token id is in `always_mask_token_ids` (e.g., `<think>`, `</think>`) get `m=1` regardless.
  - **Force-unmask**: positions where `labels == -100` get `m=0` (system/user/special embed tokens never noised).
  - `t_b' = m_b.sum() / L'` (paper Eq. 6); clamp to `1/L'` to avoid div-by-zero.
- **Test** `tests/test_noising.py`: empirical mask ratio per block ≈ `t_b` over many draws; `<think>` forced to mask; `labels=-100` positions never masked.

**3c. `diffume/diffusion/loss.py`**
- `diffusion_ce_loss(logits, labels, mask_indicator, t_prime_per_block, block_size) → scalar`:
  - For each block `b`: select positions where `mask_indicator[b] == 1` AND `labels != -100`; CE on those; sum; multiply by `1/t_b'`.
  - Average over blocks (paper Eq. 7).
- CPU-friendly pure-PyTorch; no Triton dependency.
- **Test** `tests/test_emrs_unbiased.py`: synthetic ℓ_b that depends only on `t_b'`; Monte-Carlo `E[1/t_b' · ℓ_b]` agrees within tolerance with the closed-form ideal `L*` (paper Eq. 26); same MC with `1/t_b` scaling deviates measurably (Lemma 3 bias).

---

### Phase 4 — Model wrapper + readout + InfoNCE  *(parallel; depends on 2 + 3)*

**4a. `diffume/models/readout.py`**
- `get_embedding_reps(last_hidden_state, input_ids, token_id) → (B, H)`: exact port of UME-R1 `trainer.py:568-580` — `torch.where(input_ids==token_id, arange(L), -1)` → `torch.max(dim=1)` → gather. `ValueError` on absent token (no silent fallback).
- `normalize(reps)` = `F.normalize(p=2, dim=-1)`.
- **Test** `tests/test_readout.py`: synthetic, two rows with different positions; correctness + raises.

**4b. `diffume/models/diffume_model.py`**
- `class DiffUMEModel(nn.Module)`:
  - `self.backbone`: `Qwen2_5_VLBlockDiff.from_pretrained(...)` (Phase 2b).
  - `self.proj_disc`, `self.proj_gen`: `Sequential(Linear(H,1024), GELU(), Linear(1024,1024))`.
  - `self.pbnc`: `PBNCSampler` (3a).
  - `forward_single(input_ids, labels, ...) → (diff_ce, last_hidden_state)`:
    1. `B = ceil(L / block_size)`.
    2. `t = self.pbnc.sample(B)`.
    3. `noised, mask_ind, t_prime = apply_blockwise_noise(...)` (3b).
    4. Backbone forward with `noised` + block-causal mask; capture last hidden state via hook.
    5. `diff_ce = diffusion_ce_loss(logits, labels, mask_ind, t_prime, block_size)` (3c).
    6. Return `(diff_ce, last_hidden_state)`.
  - `forward(qry_batch, pos_batch) → dict`: two `forward_single` calls; readout `<disc_emb>` and `<gen_emb>`; project + normalize; return 6 tensors.

**4c. `diffume/losses/infonce.py`**
- Re-derive `ClipLoss(local_loss=True, gather_with_grad=True)` from UME-R1 `trainer.py:449-564`. Cross-rank `torch.distributed.nn.all_gather` keeps grad. Falls back to local when not distributed.
- **Test** `tests/test_losses_shapes.py`: identical features → loss < `log(B)`; orthogonal → loss ≈ `log(B)`.

---

### Phase 5 — Dataset + collator + trainer step + smoke  *(blocking; depends on 1b + 4)*

**5a. `diffume/data/sft_dataset.py`** — `DiffUMESFTDataset`
- Wraps `load_from_disk(path)`. `__getitem__(i)` reads `qry` and `pos` sub-dicts; loads image with `process_image_unified` (UME-R1 `data_qwen.py`); applies SDAR-style chat template + `build_labels_mask`. Retry: 3 on `i`, 3 on `i+1` (UME-R1 `LazySupervisedDataset` pattern).
- **Test** `tests/test_dataset.py`: 5 rows; equal-length `input_ids`/`labels`; labels has ≥1 non-(-100); image rows have `pixel_values`.

**5b. `diffume/data/collator.py`** — `DiffUMECollator`
- Interleaves `[qry_0, pos_0, qry_1, pos_1, …]` → `(2B, L)`; pads; stacks vision tensors. Packing deferred.
- **Test** `tests/test_collator.py`: 4 pairs → `(8, max_len)`; padding mask correct.

**5c. `diffume/training/trainer_step.py`** — `compute_loss(model, batch, lambdas) → dict`
- Splits `(2B, L)` → qry/pos; calls `model.forward`;
  `loss = λ_r·(L_diff_q + L_diff_p) + λ_d·L_dctr + λ_g·L_gctr`;
  also `disc_batch_acc`, `gen_batch_acc` (in-batch top-1, `no_grad`).
- Returns 7-key dict.
- **Test** `tests/test_one_step_cpu.py`: tiny Qwen2.5-VL config (2 layers, hidden=64); full forward+backward; all 7 keys finite; `proj_disc.weight.grad` non-None non-zero; ≥1 backbone grad non-None.

**5d. `diffume/training/train.py`** — Accelerate entrypoint
- Two param groups (backbone `lr=1e-5`, heads `lr=1e-4`); `clip_grad_norm_(1.0)`; cosine + warmup; wandb; `save_state` every N steps; eval hook every M steps on a fixed 200-pair subset.
- Calls `model.pbnc.update(global_step, total_steps)` once per step.

---

### Phase 6 — Smoke run (1×GPU)  *(depends on Phase 5)*

`scripts/01_smoke_one_step.sh`:
```
accelerate launch --config_file configs/accelerate/1_gpu.yaml \
  -m diffume.training.train --config configs/diffume_v0_debug_1gpu.yaml \
  --max_steps 3 --log_every 1
```
**Pass criteria**: all 7 loss-dict keys logged for all 3 steps; no NaN/Inf; `<disc_emb>` and `<gen_emb>` found in every row; mean `t_per_block` close to PBNC mean for current step; runs in < 10 min.

Then a 10-step `torch.profiler` run → set `per_device_train_batch_size` and `gradient_accumulation_steps` in `diffume_v0_8gpu.yaml` for ≥256 effective batch.

---

### Phase 7 — 8×A100 SFT  *(depends on Phase 6)*

`scripts/10_train_8gpu.sh`: ZeRO-2, bf16, grad-ckpt, cosine + 200-step warmup.
**Monitoring gates**:
- Step 50: `L_diff_q`, `L_diff_p` finite, trending down.
- Step 200: `disc_batch_acc > 1/B`.
- Step 500: checkpoint + Phase 8 eval; require `disc_recall@1 > 3× random` to continue.

---

### Phase 8 — Inference & eval  *(parallel with 7 once first checkpoint exists)*

**8a. `diffume/inference/embed_disc.py`**: clean forward (no noise / `t_b = 0` per block), readout `<disc_emb>`, `proj_disc`, L2-norm.

**8b. `diffume/inference/embed_gen.py`**: `block_diffusion_generate` (T-step low-confidence greedy decoding), readout `<gen_emb>` after final denoise step, `proj_gen`, L2-norm.

**8c. `diffume/eval/retrieval.py`**: 1000-pair stratified sample by `dataset_name`; cosine Recall@{1,5,10}; print disc-vs-gen table per `dataset_name`. Wrapper `scripts/20_eval_retrieval.sh`.

---

## 7. Validation & sanity checks

- **Unit** (`pytest -q tests/`): tokens, prompting, dataset, collator, block-attention, PBNC, noising, EMRS-unbiased, readout, infonce — 10 test files, all CPU.
- **Integration** (`tests/test_one_step_cpu.py`): tiny Qwen2.5-VL (2 layers, hidden=64); forward+backward; all losses finite; grads non-zero on heads + ≥1 backbone param.
- **Smoke** (`scripts/01_smoke_one_step.sh`): 3 real-data steps on 1 GPU.
- **Theoretical sanity** (Phase 3c test): EMRS estimator matches the ideal NELBO (paper Lemma 4) within MC tolerance; standard `1/t_b` scaling shows measurable bias (Lemma 3).
- **Block-attention sanity** (Phase 2a test): with `block_size=1` the mask reduces to standard AR causal; with `block_size=L` the mask is fully bidirectional.
- **Decoupling**: `λ_r=0` → contrastive losses still finite (no NaN coupling).
- **Eval gate at step ~500**: `disc_recall@1 > 3× random baseline` before scaling further.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Qwen2.5-VL was trained AR — block-causal forward may catastrophically diverge at step 0 | Start PBNC from `μ_0=0.5, C_0=4` (broad, mid-range corruption); 200-step LR warmup; small `λ_r` ramp from 0.1 → 1.0 over first 1k steps if loss explodes. |
| Vision tower expects causal interaction with text — block-bidirectional may produce garbled image-text alignment | Stage-1 ablation: freeze vision tower for first 500 steps; revisit if `disc_batch_acc` stays at chance. |
| Memory: bf16 4B + 2-forward + grad-ckpt borderline on 40GB A100 | Per-rank batch=1, grad-accum to reach effective ≥256; ZeRO-2; CPU offload as fallback. |
| Block boundaries cut mid-thought → bad CE | Use `block_size=32` (small enough for diversity, large enough that within-block bidirectional context covers a whole short reasoning chunk). Paper used 32 for 4B. |
| `forward hook` capturing hidden state is forgotten between calls → memory leak | `try/finally` to `hook.remove()` after every `forward_single`. |
| InfoNCE all-gather + ZeRO-2 deadlock | UME-R1 recipe: `local_loss=True` + `gather_with_grad=True`; test on 2 GPUs first. |
| Dataset rows missing reasoning span | Phase 0b gate; `build_labels_mask` raises if assistant target span is empty. |
| New token init bad → training stalls | Mean-of-existing-rows init; optionally freeze new rows for first 100 steps. |
| Re-implementing fused diffusion CE in pure PyTorch is slow | OK for v0; CPU fallback was always the plan; Triton fused kernel deferred. |

---

## 9. Deliverables (Milestone 1)

1. `DiffUME/` standalone repo with files per §3 (no symlinks; no sys.path injection).
2. `docs/dataset_report.md` confirming dataset schema + reasoning-span coverage.
3. `pytest -q tests/` — all 10 test files pass on CPU.
4. `scripts/01_smoke_one_step.sh` — 3 steps, all 7 losses logged, no NaN/Inf.
5. One ≥1k-step 8-GPU checkpoint under `outputs/diffume-v0/`.
6. `scripts/20_eval_retrieval.sh` table: `embed_disc` vs `embed_gen` Recall@{1,5,10} on 1000-pair MMEB-V2 image subset, broken down by `dataset_name`.

---

## 10. Assumptions

- A1. Qwen2.5-VL-3B-Instruct (or 7B) HF checkpoint accessible.
- A2. `umer1_sft_processed` assistant turns contain pre-annotated `<think>…</think><answer>…</answer><gen_emb>` (gated in Phase 0b).
- A3. Single A100/H100 fits Qwen2.5-VL-3B in bf16 + grad-ckpt + BS=1 × 2-forward.
- A4. 8×A100-40GB; ZeRO-2 + bf16 fits.
- A5. `transformers==4.46.0` (UME-R1 lock) is forward-compatible with our forked Qwen2.5-VL modeling file.
- A6. Paper's PBNC defaults (`μ_0=0.5, μ_final=0.7, C_0=4, C_final=25`) generalise from LLaVA-NeXT 4B SFT to UME-R1 paired SFT (will revisit if loss curves disagree).

---

## 11. Open questions

- Q1. Image-token block placement: do we want vision tokens to live in their own block(s), or interleaved with text blocks under uniform stride? (Default: uniform stride; vision tokens have `labels=-100` so they never contribute to diff-CE.)
- Q2. Should `<think>` / `</think>` be added as new special tokens, or kept as plain text token sequences (current UME-R1 dataset has them as text)? Affects whether CoT template masking is per-token-id or per-substring.
- Q3. Vision-tower freeze — first 500 steps, or always?
- Q4. PBNC `C_final` = 25 (reasoning regime) or 50 (general regime) for embedding learning? Default 25; ablate after Phase 7.
- Q5. Block size for embeddings: 32 (paper default) vs 64 (better throughput on 16k packed seqs)?
- Q6. Hard negatives in dataset? Affects InfoNCE temperature and batch construction (Phase 0b).

---

## 12. First commands after plan approval

1. `mkdir -p DiffUME/{diffume/{data,models,diffusion,losses,training,inference,eval},configs/accelerate,scripts,tests,docs,outputs}`
2. Copy UME-R1 source files per §3.1 into `DiffUME/diffume/`.
3. Vendor `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` into `DiffUME/diffume/models/qwen2_5_vl_block_diff.py`.
4. `conda activate ride-env && python scripts/00_inspect_dataset.py --path /mnt/ceph-hdd/cold/nii00224/UME_R1/umer1_sft_processed --out docs/dataset_report.md`
5. `pytest -q tests/test_tokens.py tests/test_prompting.py`
6. After review of `docs/dataset_report.md` + Phase 0c success: implement Phases 1–5, then `bash scripts/01_smoke_one_step.sh`.
