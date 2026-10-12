#!/usr/bin/env python3
"""Test inference for the selected cropped, five-slice SegTHOR model.

Stages let Slurm run preparation/reconstruction on CPU and inference on GPU.
The raw source root must contain test/Patient_*.nii.gz. No labels are needed.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "results/segthor_full_seed43/final_exp_V2_M1_optim_seed43"
DEFAULT_REFERENCE = ROOT / "results/segthor_full_seed43/final_exp_M1_A2_unet_large"
PREPROCESSING = {"method": "hu", "windows": [[-310, 400]]}


def prepare(args, scans):
    from slice_segthor import slice_patient

    processed = args.work_dir / "data"
    marker = processed / "test_preparation.json"
    manifest = {
        "scans": [str(p.resolve()) for p in scans],
        "source_sizes": [p.stat().st_size for p in scans],
        "source_mtimes_ns": [p.stat().st_mtime_ns for p in scans],
        "shape": [256, 256], "crop_body": True, "preprocessing": PREPROCESSING,
    }
    if marker.exists():
        if json.loads(marker.read_text()) != manifest:
            raise ValueError("Test inputs changed; choose a new --work_dir")
        print(f"Reusing completed test preparation: {processed}", flush=True)
        return
    if processed.exists():
        raise ValueError(f"Incomplete preprocessing at {processed}; choose a new --work_dir")
    processed.mkdir(parents=True)
    for scan in scans:
        patient = scan.name.removesuffix(".nii.gz")
        print(f"Preparing {patient}", flush=True)
        slice_patient(patient, processed / "test", args.source_dir,
                      shape=(256, 256), test_mode=True, hu_min=-310, hu_max=400,
                      crop_body=True)
    (processed / "preprocessing.json").write_text(json.dumps(PREPROCESSING, indent=2) + "\n")
    marker.write_text(json.dumps(manifest, indent=2) + "\n")


def infer(args):
    if not (args.work_dir / "data/test_preparation.json").exists():
        raise ValueError("Run the prepare stage first")
    command = [sys.executable, str(ROOT / "ensemble.py"),
               "--models", str(args.run_dir / "model/bestmodel.pkl"),
               "--data_dir", str(args.work_dir / "data"),
               "--subset", "test", "--context_slices", "2", "--num_classes", "5",
               "--dest", str(args.work_dir / "inference"), "--save_probabilities"]
    if args.gpu:
        command.append("--gpu")
    subprocess.run(command, check=True)


def finish(args, scans):
    import nibabel as nib
    import numpy as np

    inference = args.work_dir / "inference"
    subprocess.run([
        sys.executable, str(ROOT / "ensemble.py"), "--postprocess_only",
        "--dest", str(inference), "--subset", "test",
        "--source_scan_pattern", str(args.source_dir / "test/{id_}.nii.gz"),
        "--geometry_dir", str(args.work_dir / "data/test/geometry"),
        "--postprocess", args.postprocess, "--reference_run", str(args.reference_run),
    ], check=True)
    predictions = (inference / "postprocessing/q9c/test/pred_volumes"
                   if args.postprocess == "q9c" else inference / "pred_volumes/test")
    expected = {p.name for p in scans}
    if {p.name for p in predictions.glob("*.nii.gz")} != expected:
        raise ValueError("Prediction patient list does not match the test scans")
    # Validate all volumes before publishing the submission folder.
    for scan in scans:
        source = nib.load(scan)
        prediction = nib.load(predictions / scan.name)
        labels = np.asarray(prediction.dataobj)
        if source.shape != prediction.shape or not np.allclose(source.affine, prediction.affine):
            raise ValueError(f"Prediction geometry mismatch: {scan.name}")
        if labels.dtype != np.uint8 or not np.isin(labels, [0, 1, 2, 3, 4]).all():
            raise ValueError(f"Invalid submission labels: {scan.name}")
    target = args.submission_dir / "test/pred"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".test_predictions_", dir=target.parent) as temporary:
        staged = Path(temporary) / "pred"
        staged.mkdir()
        for name in sorted(expected):
            shutil.copy2(predictions / name, staged / name)
        staged.rename(target)
    print(f"Saved {len(expected)} predictions to {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "infer", "finish", "all"], default="all")
    parser.add_argument("--source_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN,
                        help="Completed run containing model/bestmodel.pkl and model/config.json")
    parser.add_argument("--reference_run", type=Path, default=DEFAULT_REFERENCE,
                        help="Run supplying frozen anatomy and CRF settings for Q9c")
    parser.add_argument("--work_dir", type=Path, default=ROOT / "results/test_V2_seed43")
    parser.add_argument("--submission_dir", type=Path, required=True,
                        help="Group folder; predictions go into test/pred/")
    parser.add_argument("--postprocess", choices=["none", "q9c"], default="q9c")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    for field in ("source_dir", "run_dir", "reference_run", "work_dir", "submission_dir"):
        setattr(args, field, getattr(args, field).resolve())
    scans = sorted((args.source_dir / "test").glob("Patient_*.nii.gz"))
    if not scans:
        parser.error(f"No test scans in {args.source_dir / 'test'}")
    if (args.submission_dir / "test/pred").exists():
        parser.error("Submission test/pred already exists; choose a new --submission_dir")
    config = json.loads((args.run_dir / "model/config.json").read_text())
    if not (args.run_dir / "model/bestmodel.pkl").is_file():
        parser.error("Model run must contain model/bestmodel.pkl")
    if (config.get("model") != "unet-large" or config.get("context_slices") != 2
            or config.get("dataset") not in ("SEGTHOR", "SEGTHOR_CLEAN")
            or config.get("preprocessing") != PREPROCESSING):
        parser.error("This job expects the selected large U-Net, five-slice, HU [-310,400] recipe")
    if args.postprocess == "q9c":
        from ensemble import DEFAULT_CONFIG, load_config

        try:
            load_config(DEFAULT_CONFIG, args.reference_run)
        except (OSError, KeyError, ValueError) as error:
            parser.error(f"Invalid reference run: {error}")
    for stage, action in (("prepare", lambda: prepare(args, scans)),
                          ("infer", lambda: infer(args)),
                          ("finish", lambda: finish(args, scans))):
        if args.stage in (stage, "all"):
            action()


if __name__ == "__main__":
    main()
