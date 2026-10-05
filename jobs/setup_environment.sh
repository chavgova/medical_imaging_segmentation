#!/bin/bash
#SBATCH --job-name=setup_ai4mi_env
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=gpuuva084
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail

REPO="$HOME/ai4mi_project"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$REPO"

python -m venv ai4mi
source ai4mi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA build:", torch.version.cuda)'
