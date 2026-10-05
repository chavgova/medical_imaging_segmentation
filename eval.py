#!/usr/bin/env python3

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import nibabel as nib
import torch

from utils import tqdm_
from metrics import dice, hausdorff_distance_95, average_surface_distance

BACKGROUND_CLASS = 0

# SegTHOR label convention: class index -> organ name
SEGTHOR_CLASS_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]

# such that all metrics have same signature: (pred, gt, spacing, c, device)
def _dice_c(pred: np.ndarray, gt: np.ndarray, spacing, c: int = 1, device: str = "cpu") -> float:
    return float(dice(pred, gt, classes=[c])[0])


METRIC_FUNCS = {
    "dice": _dice_c,
    "hausdorff_distance_95": hausdorff_distance_95,
    "average_surface_distance": average_surface_distance,
}


def get_device(gpu: bool) -> str:
    """'cuda' if asked for and usable, else 'cpu'."""
    if not gpu:
        return "cpu"
    if not torch.cuda.is_available():
        print(">> --gpu given but CUDA is not available, evaluating on CPU")
        return "cpu"
    # Without cuCIM, MONAI computes the surface distances of CUDA tensors with SciPy on
    # CPU, which is slower than its CPU path.
    if importlib.util.find_spec("cucim") is None or importlib.util.find_spec("cupy") is None:
        print(">> --gpu given but cuCIM/CuPy is not installed, evaluating on CPU")
        return "cpu"
    return "cuda"


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D label-map volume from a .nii.gz file as a numpy array."""
    return np.asarray(nib.load(str(path)).dataobj)


def match_patients(pred_folder: Path, gt_pattern: str) -> list[str]:
    pred_ids = sorted(p.name.removesuffix(".nii.gz") for p in pred_folder.glob("*.nii.gz"))
    if not pred_ids:
        raise ValueError(f"No <patient_id>.nii.gz files found in pred_folder: {pred_folder}")

    missing_gt = [pid for pid in pred_ids if not Path(gt_pattern.format(id_=pid)).exists()]
    if missing_gt:
        raise ValueError(
            f"No ground-truth volume found (gt_pattern={gt_pattern!r}) for patients: {missing_gt}"
        )

    return pred_ids


def discover_classes(gt_paths: Sequence[Path]) -> list[int]:
    # Infer  class labels from GT volume labels.
    classes: set[int] = set()
    for path in gt_paths:
        classes |= set(np.unique(load_volume(path)).tolist())

    return sorted(classes)


def class_name(c: int, class_names: Sequence[str]) -> str:
    # Fall back to the class index for labels without a name.
    return class_names[c] if c < len(class_names) else str(c)


def evaluate_patient(patient_id: str, pred_path: Path, gt_path: Path, classes: Sequence[int], metrics: Sequence[str] = None,
                     class_names: Sequence[str] = SEGTHOR_CLASS_NAMES, device: str = "cpu") -> list[dict]:

    pred_vol = load_volume(pred_path)
    gt_vol = load_volume(gt_path)

    assert pred_vol.shape == gt_vol.shape, (
        f"Shape mismatch for patient {patient_id!r}: "
        f"pred {pred_vol.shape} vs gt {gt_vol.shape}"
    )

    spacing = nib.load(str(gt_path)).header.get_zooms()[:3]

    if metrics is None:
        metrics = ["dice", "hausdorff_distance_95", "average_surface_distance"]

    rows = []
    for c in classes:
        if c == BACKGROUND_CLASS:
            continue
        row = {"patient_id": patient_id, "class": c, "organ": class_name(c, class_names)}
        for metric in metrics:
            row[metric] = METRIC_FUNCS[metric](pred_vol, gt_vol, spacing, c=c, device=device)
        rows.append(row)

    return rows



def evaluate_dataset(pred_folder: Path, gt_pattern: str, num_classes: Optional[int] = None, metrics: Sequence[str] = None,
                     class_names: Sequence[str] = SEGTHOR_CLASS_NAMES, device: str = "cpu") -> list[dict]:

    patient_ids = match_patients(pred_folder, gt_pattern)

    if num_classes is None:
        gt_paths = [Path(gt_pattern.format(id_=pid)) for pid in patient_ids]
        classes = discover_classes(gt_paths)
    else:
        classes = list(range(num_classes))

    rows: list[dict] = []
    for pid in tqdm_(patient_ids):
        pred_path = pred_folder / f"{pid}.nii.gz"
        gt_path = Path(gt_pattern.format(id_=pid))
        rows.extend(evaluate_patient(pid, pred_path, gt_path, classes, metrics=metrics, class_names=class_names,
                                     device=device))

    return rows


def summarize(rows: Sequence[dict], metrics: Sequence[str]) -> list[dict]:

    metric_names = list(metrics)
    classes = sorted({row["class"] for row in rows})

    summary = []
    for c in classes:
        values = {name: np.array([row[name] for row in rows if row["class"] == c], dtype=np.float64)
                  for name in metric_names}

        organ = next(row["organ"] for row in rows if row["class"] == c)
        summary_row = {"class": c, "organ": organ}
        for name in metric_names:
            summary_row[f"{name}_mean"] = float(np.nanmean(values[name]))
            summary_row[f"{name}_std"] = float(np.nanstd(values[name]))

        summary.append(summary_row)

    return summary


def save_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def print_summary(summary: Sequence[dict], metrics: Sequence[str]) -> None:
    for row in summary:
        print(f"{row['organ']} (class {row['class']}): ")
        for metric in metrics:
            print(f"  {metric}: {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")


def main(args: argparse.Namespace) -> None:
    device = get_device(args.gpu)
    print(f">> Evaluating on {device}")
    rows = evaluate_dataset(args.pred_folder, args.gt_pattern, args.num_classes, args.metrics, args.class_names,
                            device)

    dest: Path = args.dest
    summary_dest = dest.with_name(f"{dest.stem}_summary{dest.suffix}")

    save_csv(rows, dest)
    print(f"Saved per-patient-per-class results to {dest}")

    summary = summarize(rows, args.metrics)
    save_csv(summary, summary_dest)
    print(f"Saved per-class summary to {summary_dest}")

    print_summary(summary, args.metrics)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation parameters")
    parser.add_argument("--pred_folder", type=Path, required=True,
                        help="Folder of predicted 3D volumes, one <patient_id>.nii.gz per patient "
                             "(e.g. stitch.py's --dest_folder output)")
    parser.add_argument("--gt_pattern", type=str, required=True,
                        help="Format-string path to each patient's ground-truth volume, with {id_} "
                             "as the patient id placeholder (same convention as stitch.py's "
                             "--source_scan_pattern). E.g. 'data/segthor_gt_val/{id_}.nii.gz' for a "
                             "flat folder, or 'data/segthor_part1/train/{id_}/GT.nii.gz' to read "
                             "directly from the raw nested layout with no copying step.")
    parser.add_argument("--metrics", nargs="*", default=["dice", "hausdorff_distance_95", "average_surface_distance"],
                        help="List of metrics to compute. Default is all.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Total number of classes, including background. "
                             "If omitted, inferred from the ground-truth volumes.")
    parser.add_argument("--class_names", type=str, nargs="+", default=SEGTHOR_CLASS_NAMES,
                        help="Organ name for each class index, starting with background (same convention "
                             "as viewer.py). Defaults to the SegTHOR labels: "
                             + " ".join(SEGTHOR_CLASS_NAMES) + ".")
    parser.add_argument("--gpu", action="store_true",
                        help="Compute HD95/ASSD on the GPU. Needs CUDA plus cuCIM and CuPy; "
                             "falls back to CPU otherwise. Dice always runs on CPU.")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Output path for the per-patient-per-class results CSV. "
                             "The per-class summary is saved alongside it as <dest>_summary.csv")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
