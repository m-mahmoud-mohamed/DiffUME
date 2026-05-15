#!/usr/bin/env bash
# Phase 5 smoke test — 1 GPU, 5 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
module load miniforge3
source "$CONDASH"
conda activate /user/mahmoud.abdellahi/u27702/.conda/envs/diffume-env
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m diffume.training.train --config configs/diffume_debug_1gpu.yaml
