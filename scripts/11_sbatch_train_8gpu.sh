#!/usr/bin/env bash
#SBATCH --job-name=diffume-stage1
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:8
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --time=47:59:00
#SBATCH --chdir=/user/mahmoud.abdellahi/u27702/diffusion/DiffUME
#SBATCH --output=/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/logs/train-stage1-%j.log
#SBATCH --error=/mnt/ceph-ssd/workspaces/ws/nii00224/u27702-diffusion-data/logs/train-stage1-%j.log

set -euo pipefail

echo "=== DiffUME Stage-1 8-GPU FSDP training ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

module load miniforge3
source "$CONDASH"
conda activate /user/mahmoud.abdellahi/u27702/.conda/envs/diffume-env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
    -m diffume.training.train --config configs/diffume_stage1.yaml
