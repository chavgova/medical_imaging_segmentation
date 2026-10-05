#!/bin/bash
#SBATCH --job-name=exp_P2_CLAHE_prep
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=gpuuva084
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail

STAGE="${1:-prepare}"
REPO="$HOME/ai4mi_project"
SCRIPT="$REPO/jobs/exp_P2_CLAHE.sh"
DATA_DIR="$REPO/data/segthor_seed43/exp_P2_CLAHE"
EXPERIMENT_DIR="$REPO/results/segthor_seed43/exp_P2_CLAHE"
RESULT_DIR="$EXPERIMENT_DIR/results"
LOG_DIR="$EXPERIMENT_DIR/logs"
SOURCE_DIR="$REPO/data/segthor_part1"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$REPO"
if [ ! -x ai4mi/bin/python ]; then
    echo "Missing ai4mi environment. Run: sbatch jobs/setup_environment.sh" >&2
    exit 1
fi
source ai4mi/bin/activate

export PYTHONHASHSEED=43
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg

case "$STAGE" in
    prepare)
        mkdir -p "$LOG_DIR"
        if [ ! -d "$DATA_DIR" ]; then
            python slice_segthor.py --source_dir "$SOURCE_DIR" --dest_dir "$DATA_DIR" \
                --shape 256 256 --retains 5 --seed 43 --fold 0 \
                --fix_aorta_esophagus --clahe \
                --process "$SLURM_CPUS_PER_TASK"
        fi
        sbatch --job-name=exp_P2_CLAHE_train --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=18 --time=04:00:00 --account=gpuuva084 \
            --output="$LOG_DIR/slurm_train_%j.out" "$SCRIPT" train
        ;;
    train)
        python main.py --dataset SEGTHOR --mode full --epochs 25 \
            --data_dir "$DATA_DIR" --dest "$RESULT_DIR" --gpu \
            --loss_fn ce --opt adam --lr 0.0005 --context_slices 0 \
            --scheduler none --deterministic --seed 43
        sbatch --job-name=exp_P2_CLAHE_eval --partition=rome --cpus-per-task=16 \
            --time=01:00:00 --account=gpuuva084 \
            --output="$LOG_DIR/slurm_eval_%j.out" "$SCRIPT" evaluate
        ;;
    evaluate)
        python stitch.py --data_folder "$RESULT_DIR/best_epoch/val" \
            --dest_folder "$RESULT_DIR/pred_volumes" --num_classes 255 \
            --grp_regex '(Patient_[0-9]+)_[0-9]+' \
            --source_scan_pattern "$SOURCE_DIR/train/{id_}/GT.nii.gz"
        python stitch.py --data_folder "$DATA_DIR/val/gt" \
            --dest_folder "$RESULT_DIR/gt_volumes" --num_classes 255 \
            --grp_regex '(Patient_[0-9]+)_[0-9]+' \
            --source_scan_pattern "$SOURCE_DIR/train/{id_}/GT.nii.gz"
        python eval.py --pred_folder "$RESULT_DIR/pred_volumes" \
            --gt_pattern "$RESULT_DIR/gt_volumes/{id_}.nii.gz" --num_classes 5 \
            --metrics dice hausdorff_distance_95 average_surface_distance \
            --dest "$RESULT_DIR/metrics.csv"
        python plot.py --metric_file "$RESULT_DIR/dice_tra.npy" \
            --dest "$RESULT_DIR/dice_train.png" --headless
        python plot.py --metric_file "$RESULT_DIR/dice_val.npy" \
            --dest "$RESULT_DIR/dice_validation.png" --headless
        python plot.py --metric_file "$RESULT_DIR/loss_tra.npy" \
            --dest "$RESULT_DIR/loss_train.png" --headless
        python plot.py --metric_file "$RESULT_DIR/loss_val.npy" \
            --dest "$RESULT_DIR/loss_validation.png" --headless
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac
