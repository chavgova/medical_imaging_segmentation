#!/bin/bash
#SBATCH --job-name=exp_T1_lr0p0001_submit
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --account=gpuuva084
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail

STAGE="${1:-submit}"
REPO="$HOME/ai4mi_project"
SCRIPT="$REPO/jobs/exp_T1_lr0p0001.sh"
DATA_DIR="$REPO/data/segthor_seed43/exp_P1_HU"
EXPERIMENT_DIR="$REPO/results/segthor_seed43/exp_T1_lr0p0001"
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
    submit)
        if [ ! -d "$DATA_DIR/train/img" ] || [ ! -d "$DATA_DIR/val/img" ]; then
            echo "Missing P1 data: $DATA_DIR" >&2
            exit 1
        fi
        mkdir -p "$LOG_DIR"
        sbatch --job-name=exp_T1_lr0p0001_train --partition=gpu_a100 --gpus=1 \
            --cpus-per-task=18 --time=04:00:00 --account=gpuuva084 \
            --output="$LOG_DIR/slurm_train_%j.out" "$SCRIPT" train
        ;;
    train)
        python main.py --dataset SEGTHOR --mode full --epochs 25 \
            --data_dir "$DATA_DIR" --dest "$RESULT_DIR" --gpu \
            --loss_fn balance --balance_alpha 0.5 --balance_t 0.9 \
            --balance_fallback_epoch -1 --balance_normalized \
            --opt adam --lr 0.0001 --context_slices 2 \
            --scheduler none --deterministic --seed 43 --augment
        sbatch --job-name=exp_T1_lr0p0001_eval --partition=rome --cpus-per-task=16 \
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
