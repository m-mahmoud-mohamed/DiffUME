# DiffUME — Reasoning-Driven Block-wise Diffusion Multimodal Embeddings

DiffUME converts the [UME-R1-2B](https://huggingface.co/zhibinlan/UME-R1-2B) vision-language model from autoregressive generation into a **block-wise discrete diffusion embedding model**.  
It applies the SDAR-VL recipe — **BD³ + ABNS + EMRS + PBNC** — directly on top of the Qwen2-VL backbone, producing multimodal embeddings that are grounded in multi-step visual reasoning chains.

---

## How it works

```
[system + question + image tokens] │ [CoT block 0] │ [CoT block 1] │ … │ [answer] │ <disc_emb> <gen_emb>
                                   └─────────────────── diffusable span (block-wise noised) ──────────────┘
```

1. **Block-wise noising** — the assistant span is divided into blocks of 4 tokens. Each block is independently masked at noise level $t_b$ sampled by the PBNC beta-noise scheduler (μ = 0.8, C ramps from 2 → 50 during warmup).
2. **Block-causal attention** — a 4D additive mask lets each token attend to all prior blocks but only revealed tokens within its own block, enabling iterative denoising while preserving causal order.
3. **Dual readout heads** — two special tokens appended to every sequence produce:
   - `z_disc` — L2-normalised discrete/retrieval embedding (contrastive loss)
   - `z_gen`  — L2-normalised generative embedding
4. **Full backbone fine-tuning** — all 2.2 B Qwen2-VL parameters are trained end-to-end through the diffusion objective (FSDP full-shard, bf16). No new layers are inserted inside the transformer; diffusion control is applied externally via the 4D attention mask and position IDs, keeping the architecture clean and the HuggingFace implementation unmodified.

---

## Architecture overview

| Component | Details |
|---|---|
| Backbone | `zhibinlan/UME-R1-2B` — Qwen2-VL, 2.2 B params, hidden 1536, 28 layers |
| Vision encoder | ViT (32 × ViT blocks) + MLP merger, 3D RoPE M-RoPE |
| Diffusion block size | 4 tokens per block |
| Noise scheduler | PBNC Beta (μ=0.8, C: 2→50, warmup 8 090 steps) |
| Loss | Diffusion CE (BD³) + EMRS disc/gen contrastive |
| Precision | bf16, FSDP full-shard (`Qwen2VLDecoderLayer`) |

---

## Repository layout

```
DiffUME/
├── configs/
│   ├── diffume_stage1.yaml          # 8-GPU full SFT config
│   ├── diffume_debug_1gpu.yaml      # 1-GPU smoke (5 steps)
│   └── accelerate/
│       ├── fsdp_8gpu.yaml
│       └── single_gpu.yaml
├── diffume/
│   ├── data/                        # dataset + collation
│   ├── diffusion/                   # noising, PBNC scheduler, loss
│   ├── models/                      # DiffumeModel wrapper, block-causal mask
│   ├── losses/                      # BD³ + EMRS loss functions
│   ├── training/                    # train.py entrypoint, optimizer, utils
│   ├── eval/                        # retrieval evaluation
│   └── inference/                   # clean-encode (embed_disc) path
├── scripts/
│   ├── 00_inspect_dataset.py
│   ├── 01_smoke_one_step.sh         # 1-GPU, 5 steps
│   ├── 10_train_8gpu.sh             # interactive 8-GPU run
│   └── 11_sbatch_train_8gpu.sh      # SLURM submission
└── tests/                           # CPU-only unit tests (pytest)
```

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Unit tests (CPU-only, ~10 s)
pytest -q tests/

# 3. Smoke run (1 GPU, 5 steps)
bash scripts/01_smoke_one_step.sh

# 4. Full SFT — interactive (8 × A100, run from a compute node)
bash scripts/10_train_8gpu.sh

# 5. Full SFT — SLURM submission
sbatch scripts/11_sbatch_train_8gpu.sh
```

---

## Training config highlights

| Knob | Value |
|---|---|
| GPUs | 8 × A100-80 GB |
| Effective batch size | 64 (1 × 8 GPUs × 8 grad-accum) |
| Max sequence length | 1 536 tokens |
| Learning rate | 1e-5, cosine with 500-step warmup |
| Max steps | 30 000 |
| Save every | 2 000 steps |

---

## Key dependencies

| Package | Min version |
|---|---|
| PyTorch | 2.1 |
| Transformers | 4.49.0 |
| Accelerate | 0.34 |
| DeepSpeed | 0.14 |
| Python | 3.10 |

---

## References

- SDAR-VL: *Score-based Discrete Autoregressive Reasoning for VLMs* (BD³ + PBNC recipe)
- UME-R1: `zhibinlan/UME-R1-2B` — reasoning-finetuned Qwen2-VL-2B
- Qwen2-VL: *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*
