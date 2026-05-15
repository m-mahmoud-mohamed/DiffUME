#!/usr/bin/env bash
# Phase 5 full SFT — 8 GPU, DeepSpeed ZeRO-2, bf16.
set -euo pipefail
cd "$(dirname "$0")/.."
module load miniforge3
source "$CONDASH"
conda activate /user/mahmoud.abdellahi/u27702/.conda/envs/diffume-env
accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
    -m diffume.training.train --config configs/diffume_stage1.yaml
