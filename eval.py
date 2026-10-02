#!/usr/bin/env python3

import argparse
import csv
import json
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import nibabel as nib

from utils import tqdm_
from metrics import dice, hausdorff_distance_95, average_surface_distance
from postprocessing import (anatomy_aware_filtering, closing, fill_holes,
                            keep_largest_connected_components, opening,
                            salt_and_pepper)

BACKGROUND_CLASS = 0

# Add future techniques here; each callable takes and returns a 3D label map.
POSTPROCESSORS = {
    "largest_connected_components": keep_largest_connected_components,
    "fill_holes": fill_holes,
    "opening": opening,
    "closing": closing,
    "salt_and_pepper": salt_and_pepper,
}
CONFIGURED_POSTPROCESSORS = {"anatomy_aware_filtering": anatomy_aware_filtering}
POSTPROCESSOR_NAMES = [*POSTPROCESSORS, *CONFIGURED_POSTPROCESSORS]

class PostprocessingPipeline:
    """applies ordered post-processing steps with per-patient spatial context"""

    def __init__(self, steps: Sequence[tuple[Callable, bool]]):
        self.steps = list(steps)

    def __call__(self, volume: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
        result = volume
        for processor, needs_spacing in self.steps:
            result = processor(result, spacing=spacing) if needs_spacing else processor(result)
        return result

# such that all metrics have same signature: (pred, gt, spacing, c)
def _dice_c(pred: np.ndarray, gt: np.ndarray, spacing, c: int = 1) -> float:
    return float(dice(pred, gt, classes=[c])[0])


METRIC_FUNCS = {
    "dice": _dice_c,
    "hausdorff_distance_95": hausdorff_distance_95,
    "average_surface_distance": average_surface_distance,
}


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


def evaluate_patient(patient_id: str, pred_path: Path, gt_path: Path, classes: Sequence[int], metrics: Sequence[str] = None,
                     postprocess: Optional[Callable[[np.ndarray], np.ndarray] | PostprocessingPipeline] = None,
                     save_folder: Optional[Path] = None) -> list[dict]:

    save_path = None
    if save_folder is not None:
        save_path = Path(save_folder) / pred_path.name
        for source in (pred_path, gt_path):
            if (save_path.resolve() == source.resolve()
                    or (save_path.exists() and save_path.samefile(source))):
                raise ValueError(f"Cannot overwrite an input volume: {save_path}")

    pred_image = nib.load(str(pred_path))
    pred_vol = np.asarray(pred_image.dataobj)
    gt_vol = load_volume(gt_path)

    assert pred_vol.shape == gt_vol.shape, (
        f"Shape mismatch for patient {patient_id!r}: "
        f"pred {pred_vol.shape} vs gt {gt_vol.shape}"
    )

    spacing = nib.load(str(gt_path)).header.get_zooms()[:3]

    if postprocess is not None:
        if isinstance(postprocess, PostprocessingPipeline):
            pred_vol = postprocess(pred_vol, spacing)
        else:
            pred_vol = postprocess(pred_vol)

    if metrics is None:
        metrics = ["dice", "hausdorff_distance_95", "average_surface_distance"]

    rows = []
    for c in classes:
        if c == BACKGROUND_CLASS:
            continue
        row = {"patient_id": patient_id, "class": c}
        for metric in metrics:
            row[metric] = METRIC_FUNCS[metric](pred_vol, gt_vol, spacing, c=c)
        rows.append(row)

    if save_path is not None:
        # Keep prediction geometry; filtering does not resample or align voxels.
        saved_image = pred_image.__class__(pred_vol, pred_image.affine, header=pred_image.header.copy())
        saved_image.set_qform(*pred_image.get_qform(coded=True))
        saved_image.set_sform(*pred_image.get_sform(coded=True))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(saved_image, str(save_path))

    return rows



def evaluate_dataset(pred_folder: Path, gt_pattern: str, num_classes: Optional[int] = None, metrics: Sequence[str] = None,
                     postprocess: Optional[Callable[[np.ndarray], np.ndarray] | PostprocessingPipeline] = None,
                     save_folder: Optional[Path] = None) -> list[dict]:

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
        rows.extend(evaluate_patient(pid, pred_path, gt_path, classes, metrics=metrics,
                                     postprocess=postprocess, save_folder=save_folder))

    return rows


def summarize(rows: Sequence[dict], metrics: Sequence[str]) -> list[dict]:

    metric_names = list(metrics)
    classes = sorted({row["class"] for row in rows})

    summary = []
    for c in classes:
        values = {name: np.array([row[name] for row in rows if row["class"] == c], dtype=np.float64)
                  for name in metric_names}

        summary_row = {"class": c}
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
        print(f"Class {row['class']}: ")
        for metric in metrics:
            print(f"  {metric}: {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")


def build_postprocessing_pipeline(args: argparse.Namespace) -> Optional[PostprocessingPipeline]:
    if args.postprocessing == ["none"]:
        return None

    configured = {}
    if "anatomy_aware_filtering" in args.postprocessing:
        try:
            with open(args.postprocessing_config) as config_file:
                root_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load post-processing config {args.postprocessing_config}: {error}") from error
        configured = root_config.get("anatomy_aware_filtering")

    steps: list[tuple[Callable, bool]] = []
    for method in args.postprocessing:
        if method == "anatomy_aware_filtering":
            processor = partial(anatomy_aware_filtering, config=configured)
            steps.append((processor, True))
            continue

        options = {"classes": args.postprocessing_classes}
        if args.connectivity is not None and method != "salt_and_pepper":
            options["connectivity"] = args.connectivity
        if method == "largest_connected_components":
            options["k"] = args.top_k
        if method in ("opening", "closing"):
            options["iterations"] = args.iterations
        if method == "salt_and_pepper":
            options["kernel_size"] = args.kernel_size
        steps.append((partial(POSTPROCESSORS[method], **options), False))

    return PostprocessingPipeline(steps)


def main(args: argparse.Namespace) -> None:
    postprocess = build_postprocessing_pipeline(args)
    save_folder = None
    if args.save:
        save_folder = args.save_folder or args.dest.parent / f"{args.dest.stem}_volumes"
    rows = evaluate_dataset(args.pred_folder, args.gt_pattern, args.num_classes, args.metrics,
                            postprocess=postprocess, save_folder=save_folder)
    if save_folder is not None:
        print(f"Saved evaluated prediction volumes to {save_folder}")

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
    parser.add_argument("--postprocessing", choices=["none", *POSTPROCESSOR_NAMES], nargs="+", default=["none"],
                        help="Ordered techniques applied in memory before scoring (default: none)")
    parser.add_argument("--postprocessing_config", type=Path,
                        default=Path(__file__).with_name("postprocessing_config.json"),
                        help="JSON parameters for configured methods such as anatomy_aware_filtering")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of components per class for largest_connected_components only (default: 1).")
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=None,
                        help="3D neighborhood / morphology structuring element. "
                             "Defaults: 26 for largest_connected_components, 6 for fill_holes/opening/closing.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Positive number of erosion/dilation steps per stage for opening/closing only (default: 1).")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Odd positive cubic window width for salt_and_pepper only (default: 3 voxels).")
    parser.add_argument("--postprocessing_classes", type=int, nargs="+", default=None,
                        help="Labels to filter; defaults to all nonzero prediction labels.")
    parser.add_argument("--save", action="store_true",
                        help="Save evaluated predictions as .nii.gz files after optional post-processing.")
    parser.add_argument("--save_folder", type=Path, default=None,
                        help="Output volume folder (requires --save). Default: <dest stem>_volumes "
                             "beside the results CSV. Existing output files are replaced.")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Output path for the per-patient-per-class results CSV. "
                             "The per-class summary is saved alongside it as <dest>_summary.csv")

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be a positive integer")
    if args.iterations < 1:
        parser.error("--iterations must be a positive integer")
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        parser.error("--kernel_size must be a positive odd integer")
    if "none" in args.postprocessing and args.postprocessing != ["none"]:
        parser.error("--postprocessing none cannot be combined with another method")
    if len(args.postprocessing) != len(set(args.postprocessing)):
        parser.error("--postprocessing methods cannot be repeated")
    if args.postprocessing == ["salt_and_pepper"] and args.connectivity is not None:
        parser.error("salt_and_pepper uses --kernel_size, not --connectivity")
    if args.save_folder is not None and not args.save:
        parser.error("--save_folder requires --save")

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
