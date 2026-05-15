#!/usr/bin/env bash
#SBATCH --job-name=diffume-smoke
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --chdir=/user/mahmoud.abdellahi/u27702/diffusion/DiffUME
#SBATCH --output=/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/logs/smoke-%j.log
#SBATCH --error=/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/logs/smoke-%j.log

set -euo pipefail

module load miniforge3
source "$CONDASH"
conda activate /user/mahmoud.abdellahi/u27702/.conda/envs/diffume-env

echo "=== DiffUME smoke test ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m diffume.training.train --config configs/diffume_debug_1gpu.yaml
