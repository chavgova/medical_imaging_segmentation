#!/bin/bash
#SBATCH --job-name=segthor_test_prepare
#SBATCH --partition=rome
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=gisr128855
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/ai4mi_project}}"
export MODEL_RUN="${MODEL_RUN:-$REPO/results/segthor_full_seed43/final_exp_V2_M1_optim_seed43}"
export REFERENCE_RUN="${REFERENCE_RUN:-$REPO/results/segthor_full_seed43/final_exp_M1_A2_unet_large}"
export SOURCE_DIR="${SOURCE_DIR:-$REPO/data}"
export WORK_DIR="${WORK_DIR:-$REPO/results/test_V2_seed43}"
export SUBMISSION_DIR="${SUBMISSION_DIR:?Set SUBMISSION_DIR to your group folder, e.g. $REPO/group-XX}"
export POSTPROCESS="${POSTPROCESS:-q9c}"
SCRIPT="$REPO/jobs/predict_test.sh"

module load 2023 Python/3.11.3-GCCcore-12.3.0
cd "$REPO"
source ai4mi/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

ARGS=(--source_dir "$SOURCE_DIR" --run_dir "$MODEL_RUN" --reference_run "$REFERENCE_RUN"
      --work_dir "$WORK_DIR" --submission_dir "$SUBMISSION_DIR" --postprocess "$POSTPROCESS")

case "${1:-prepare}" in
    prepare)
        python predict_test.py "${ARGS[@]}" --stage prepare
        sbatch --job-name=segthor_test_infer --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=10 --mem=32G --time=02:00:00 \
            --output="$REPO/logs/slurm_%x_%j.out" "$SCRIPT" infer
        ;;
    infer)
        python predict_test.py "${ARGS[@]}" --stage infer --gpu
        sbatch --job-name=segthor_test_finish --partition=rome --cpus-per-task=10 \
            --mem=64G --time=06:00:00 \
            --output="$REPO/logs/slurm_%x_%j.out" "$SCRIPT" finish
        ;;
    finish)
        python predict_test.py "${ARGS[@]}" --stage finish
        ;;
    *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
