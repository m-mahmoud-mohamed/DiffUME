# DiffUME

Reasoning-Driven Block-wise Diffusion Multimodal Embeddings.

Applies the SDAR-VL recipe (BD3 + ABNS + EMRS + PBNC) directly on top of
UME-R1's Qwen2.5-VL stack — converting it from autoregressive embeddings
to **block-wise discrete diffusion embeddings**.

See [plan-diffumeReasoningDiffusionEmbeddings.prompt.md](plan-diffumeReasoningDiffusionEmbeddings.prompt.md)
for the full design and roadmap.

## Quick start

```bash
conda activate ride-env
pip install -e .

# Phase 0b — dataset audit (required before training):
python scripts/00_inspect_dataset.py \
    --path /mnt/ceph-hdd/cold/nii00224/UME_R1/umer1_sft_processed \
    --out docs/dataset_report.md

# Unit tests (CPU-only):
pytest -q tests/

# Smoke (1 GPU, 3 steps):
bash scripts/01_smoke_one_step.sh

# Full SFT (8 × A100):
bash scripts/10_train_8gpu.sh
```
