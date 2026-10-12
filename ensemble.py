#!/usr/bin/env python3
"""Ensemble inference from full PyTorch models in one or more locations.

Use --ensemble_dir for one folder, or --models for a list of checkpoint files
and/or separate run folders (including runs with a model/ subfolder).

By default, find *.pkl models recursively (including training's bestmodel.pkl).
For other filenames, pass --model_pattern, e.g. '**/*.pt'. These must be
trusted full models saved with torch.save(model, path), not state dictionaries.
Members must share preprocessing, input channels and output class ordering.
Uses SliceDataset and transforms matching main.py; train/val subsets require ground
truth PNGs, while the test subset only requires input images.

To rerun reconstruction/Q9c without inference, pass --postprocess_only,
--dest, --source_scan_pattern, --geometry_dir and --postprocess q9c.

Outputs:
  predictions/<subset>/<stem>.png: same label encoding as main.py.
  uncertainty/<subset>/<stem>.npz: normalized entropy (H, W), normalized
    mutual_information (H, W), and population variance per class (K, H, W).
  With --save_probabilities, --export_volumes or --postprocess q9c:
    probabilities/<subset>/<stem>.npz: averaged softmax probabilities (K, H, W).
  With --export_volumes or --postprocess q9c:
    pred_volumes/<subset>/*.nii.gz: raw labels on the original CT grid.
    probability_volumes/<subset>/*.nii.gz: native probabilities (X, Y, Z, K).
  With --postprocess q9c:
    postprocessing/q9c/<subset>/pred_volumes/*.nii.gz: final Q9c labels.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from functools import partial
from copy import deepcopy

from utils import save_images, tqdm_, class2one_hot



def img_transform(img):
    """Match main.py's image transform without importing training dependencies."""
    values = np.array(img.convert("L"))[np.newaxis, ...]
    return torch.tensor(values / 255, dtype=torch.float32)


def gt_transform(class_count, img):
    """Match main.py's label encoding and one-hot transform."""
    values = np.array(img)
    values = values / (63 if class_count == 5 else 255 / (class_count - 1))
    return class2one_hot(torch.tensor(values, dtype=torch.int64)[None, ...], K=class_count)[0]


DEFAULT_CONFIG = Path(__file__).with_name("postprocessing_config.json")


def reference_config_path(reference_run, method):
    reference = Path(reference_run) / "postprocessing"
    candidates = [reference / method / "postprocessing_config.json"]
    if method == "anatomy_aware_filtering":
        candidates.append(reference / "crf_aorta_anatomy12_fill" / "postprocessing_config.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing frozen {method} config; checked: " + ", ".join(str(p) for p in candidates))


def load_config(config_path, reference_run=None):
    config = json.loads(Path(config_path).read_text())
    if reference_run is not None:
        for method in ("anatomy_aware_filtering", "dense_crf"):
            frozen = json.loads(reference_config_path(reference_run, method).read_text())
            config[method] = frozen[method]
    config = deepcopy(config)
    config["anatomy_aware_filtering"]["classes"] = {
        key: rule for key, rule in config["anatomy_aware_filtering"]["classes"].items()
        if key in ("1", "2")
    }
    config["combination"] = {
        "steps": ["crf_aorta", "anatomy12", "fill_holes"],
        "crf_editable_labels": [0, 4],
        "fill_holes": {"classes": [2], "connectivity": 6},
    }
    return config


def apply_q9c(raw, probabilities, image, spacing, config):
    """Match Q9c: CRF aorta transfer, anatomy classes 1/2, heart holes."""
    import postprocessing as pp

    crf = pp.dense_crf(probabilities.argmax(-1).astype(np.uint8),
                       probabilities, image, spacing, config["dense_crf"])
    labels = raw.copy()
    editable = (raw == 0) | (raw == 4)
    labels[editable] = np.where(crf[editable] == 4, 4, 0)
    anatomy = deepcopy(config["anatomy_aware_filtering"])
    anatomy["classes"] = {key: rule for key, rule in anatomy["classes"].items()
                          if key in ("1", "2")}
    labels = pp.anatomy_aware_filtering(labels, spacing, anatomy)
    return pp.fill_holes(labels, classes=[2], connectivity=6)


def restore_patient(paths, source, geometry_path, class_count=5):
    """Restore raw nearest-neighbor labels and linearly resized probabilities."""
    from skimage.transform import resize

    if len(source.shape) != 3:
        raise ValueError("Original CT must be three-dimensional")
    x, y, z = source.shape
    geometry = json.loads(Path(geometry_path).read_text())
    if geometry["original_shape"] != list(source.shape):
        raise ValueError(f"Geometry shape mismatch: {geometry_path}")
    if geometry.get("target_spacing") is not None:
        raise ValueError("Spacing resampling is not supported for reconstruction")
    r0, r1, c0, c1 = geometry["crop_bbox"]
    if not (0 <= r0 < r1 <= x and 0 <= c0 < c1 <= y):
        raise ValueError(f"Invalid crop bounds: {geometry_path}")
    if sorted(paths) != list(range(z)):
        raise ValueError(f"Expected all {z} CT slices, found {sorted(paths)}")
    probabilities = np.zeros((x, y, z, class_count), dtype=np.float32)
    probabilities[..., 0] = 1.0
    raw = np.zeros((x, y, z), dtype=np.uint8)
    shape = (r1 - r0, c1 - c0)
    for index, path in sorted(paths.items()):
        with np.load(path) as saved:
            slab = saved["probabilities"]
        if (slab.ndim != 3 or slab.shape[0] != class_count
                or not np.isfinite(slab).all() or slab.min() < 0 or slab.max() > 1
                or not np.allclose(slab.sum(0), 1, atol=1e-5)):
            raise ValueError(f"Invalid class probabilities: {path}")
        restored = np.stack([
            resize(channel, shape, order=1, mode="constant",
                   preserve_range=True, anti_aliasing=False)
            for channel in slab
        ], axis=-1).astype(np.float32)
        restored /= restored.sum(-1, keepdims=True)
        probabilities[r0:r1, c0:c1, index, :] = restored
        raw[r0:r1, c0:c1, index] = resize(
            slab.argmax(0), shape, order=0, mode="constant",
            preserve_range=True, anti_aliasing=False).astype(np.uint8)
    return raw, probabilities


def save_volume(array, source, path):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    header = source.header.copy()
    header.set_data_dtype(array.dtype)
    nib.save(nib.Nifti1Image(array, source.affine, header), path)


def process_ensemble(dest, subset, source_scan_pattern, geometry_dir,
                     method="none", config_path=DEFAULT_CONFIG, reference_run=None,
                     stems=None):
    import nibabel as nib

    dest, geometry_dir = Path(dest), Path(geometry_dir)
    if "{id_}" not in source_scan_pattern:
        raise ValueError("--source_scan_pattern must contain {id_}")
    probability_dir = dest / "probabilities" / subset
    paths = ([probability_dir / f"{stem}.npz" for stem in stems] if stems is not None
             else sorted(probability_dir.glob("*.npz")))
    if not paths:
        raise ValueError(f"No saved ensemble probabilities in {probability_dir}")
    patients = {}
    for path in paths:
        patient, z = path.stem.rsplit("_", 1)
        slices = patients.setdefault(patient, {})
        index = int(z)
        if index in slices:
            raise ValueError(f"Duplicate slice: {path}")
        slices[index] = path
    config = load_config(config_path, reference_run) if method == "q9c" else None
    if method == "q9c":
        try:
            import pydensecrf.densecrf  # Check before writing large volumes.
        except ImportError as exc:
            raise ImportError("Q9c requires pydensecrf; install the project's requirements.txt") from exc
    for patient, slices in sorted(patients.items()):
        source = nib.load(source_scan_pattern.format(id_=patient))
        raw, probabilities = restore_patient(slices, source, geometry_dir / f"{patient}.json")
        save_volume(raw, source, dest / "pred_volumes" / subset / f"{patient}.nii.gz")
        save_volume(probabilities, source, dest / "probability_volumes" / subset / f"{patient}.nii.gz")
        if method == "q9c":
            print(f">> Q9c: {patient}", flush=True)
            labels = apply_q9c(raw, probabilities, np.asarray(source.dataobj),
                               source.header.get_zooms()[:3], config)
            output = dest / "postprocessing" / "q9c" / subset
            save_volume(labels, source, output / "pred_volumes" / f"{patient}.nii.gz")
        print(f">> Restored ensemble volumes: {patient}", flush=True)
    if config is not None:
        config["dense_crf"]["image_pattern"] = source_scan_pattern
        config["dense_crf"]["probability_pattern"] = str(dest / "probability_volumes" / subset / "{id_}.nii.gz")
        config["geometry_dir"] = str(geometry_dir)
        config["reference_run"] = str(reference_run) if reference_run is not None else None
        (dest / "postprocessing" / "q9c" / subset / "postprocessing_config.json").write_text(
            json.dumps(config, indent=2) + "\n")



def find_model_paths(locations, pattern: str):
    """Resolve files/folders and avoid weighting duplicate checkpoints twice."""
    paths = []
    seen = set()
    for location in locations:
        location = Path(location).expanduser()
        if location.is_file():
            candidates = [location]
        elif location.is_dir():
            candidates = sorted(path for path in location.glob(pattern) if path.is_file())
            if not candidates:
                raise ValueError(f"No models matching {pattern!r} in {location}")
        else:
            raise ValueError(f"Model file or folder does not exist: {location}")
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    if not paths:
        raise ValueError("At least one model file or folder is required")
    return paths


def load_models(folder: Path, pattern: str, device: torch.device,
                context_slices: int, class_count: int, model_locations=None):
    paths = find_model_paths(model_locations if model_locations is not None else [folder], pattern)
    models = []
    for path in paths:
        config_path = path.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if int(config.get("context_slices", 0)) != context_slices:
                raise ValueError(f"{path}: context_slices differs from --context_slices")
            dataset_classes = {"TOY2": 2, "SEGTHOR": 5, "SEGTHOR_CLEAN": 5}
            if config.get("dataset") in dataset_classes and dataset_classes[config["dataset"]] != class_count:
                raise ValueError(f"{path}: class count differs from --num_classes")
        model = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(model, nn.Module):
            raise ValueError(f"{path}: expected a full model saved with torch.save(model, path); "
                             "use bestmodel.pkl rather than bestweights.pt")
        models.append(model.to(device).eval())
    return models, paths


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny).log()).sum(dim=1)


@torch.inference_mode()
def predict_ensemble(models, images, class_count):
    """Average probabilities, computing uncertainty with streaming moments."""
    if class_count < 2:
        raise ValueError("At least two classes are required")
    mean = squared_deviations = entropy_sum = None
    expected_shape = (images.shape[0], class_count, *images.shape[2:])
    for count, model in enumerate(models, start=1):
        logits = model(images)
        if logits.shape != expected_shape:
            raise ValueError(f"Model output {tuple(logits.shape)} must match {expected_shape}")
        probabilities = logits.float().softmax(dim=1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Model produced nonfinite probabilities")
        if mean is None:
            mean = probabilities.clone()
            squared_deviations = torch.zeros_like(mean)
            entropy_sum = entropy(probabilities)
        else:
            delta = probabilities - mean
            mean += delta / count
            squared_deviations += delta * (probabilities - mean)
            entropy_sum += entropy(probabilities)
    if mean is None:
        raise ValueError("Ensemble must contain at least one model")
    predictive_entropy = entropy(mean)
    uncertainty = {
        "entropy": (predictive_entropy / math.log(class_count)).clamp(0, 1),
        "mutual_information": ((predictive_entropy - entropy_sum / count) /
                               math.log(class_count)).clamp(0, 1),
        "variance": (squared_deviations / count).clamp_min(0),
    }
    return mean, uncertainty


def run(args):
    if args.postprocess_only:
        process_ensemble(args.dest, args.subset, args.source_scan_pattern, args.geometry_dir,
                         args.postprocess, args.postprocessing_config, args.reference_run)
        return
    from dataset import SliceDataset

    export_volumes = args.export_volumes or args.postprocess == "q9c"
    save_probabilities = args.save_probabilities or export_volumes
    if args.postprocess == "q9c":
        try:
            import pydensecrf.densecrf
        except ImportError as exc:
            raise ImportError("Q9c requires pydensecrf; install the project's requirements.txt") from exc
        load_config(args.postprocessing_config, args.reference_run)
    if args.batch_size < 1 or args.context_slices < 0 or args.num_classes < 2:
        raise ValueError("Require batch_size > 0, context_slices >= 0 and num_classes >= 2")
    if args.gpu:
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.mps:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    models, paths = load_models(args.ensemble_dir, args.model_pattern, device,
                               args.context_slices, args.num_classes,
                               model_locations=args.models)
    dataset = SliceDataset(
        args.subset, args.data_dir, img_transform=img_transform,
        gt_transform=partial(gt_transform, args.num_classes),
        context_slices=args.context_slices)
    if not len(dataset):
        raise ValueError(f"No images in {args.data_dir / args.subset / 'img'}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prediction_dir = args.dest / "predictions" / args.subset
    uncertainty_dir = (args.uncertainty_dest or args.dest / "uncertainty") / args.subset
    for directory in (prediction_dir, uncertainty_dir):
        directory.mkdir(parents=True, exist_ok=True)
    probability_dir = args.dest / "probabilities" / args.subset
    if save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    print(f">> Ensemble of {len(models)} models on {device}: {[str(path) for path in paths]}")
    for batch in tqdm_(loader):
        mean, maps = predict_ensemble(models, batch["images"].to(device), args.num_classes)
        multiplier = 63 if args.num_classes == 5 else 255 / (args.num_classes - 1)
        save_images(mean.argmax(dim=1) * multiplier, batch["stems"], prediction_dir)
        maps = {key: value.cpu().numpy() for key, value in maps.items()}
        probabilities = mean.cpu().numpy() if save_probabilities else None
        for index, stem in enumerate(batch["stems"]):
            stems.append(stem)
            np.savez_compressed(uncertainty_dir / f"{stem}.npz",
                                **{key: value[index] for key, value in maps.items()})
            if save_probabilities:
                np.savez_compressed(probability_dir / f"{stem}.npz",
                                    probabilities=probabilities[index])
    manifest = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key != "models"}
    manifest["models"] = [str(path.resolve()) for path in paths]
    manifest["save_probabilities"] = save_probabilities
    manifest["export_volumes"] = export_volumes
    (args.dest / f"ensemble_config_{args.subset}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f">> Predictions: {prediction_dir}; uncertainty: {uncertainty_dir}")
    if export_volumes:
        process_ensemble(args.dest, args.subset, args.source_scan_pattern, args.geometry_dir,
                         args.postprocess, args.postprocessing_config, args.reference_run,
                         stems=stems)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--ensemble_dir", "--ensemble", type=Path,
                         help="Folder containing ensemble checkpoints (searched recursively)")
    sources.add_argument("--postprocess_only", action="store_true",
                         help="Restore/postprocess saved probabilities without loading models")
    sources.add_argument("--models", type=Path, nargs="+",
                         help="Checkpoint files and/or run folders; folders use --model_pattern")
    parser.add_argument("--model_pattern", default="**/*.pkl")
    parser.add_argument("--data_dir", type=Path, help="Processed data root")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--uncertainty_dest", type=Path)
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--context_slices", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_probabilities", action="store_true",
                        help="Save averaged probabilities for later reconstruction/postprocessing")
    parser.add_argument("--export_volumes", action="store_true",
                        help="Restore raw labels and averaged probabilities to native CT volumes")
    parser.add_argument("--postprocess", choices=("none", "q9c"), default="none",
                        help="Q9c also enables probability saving and native volume export")
    parser.add_argument("--source_scan_pattern", help="Original CT path containing {id_}; required for volume export")
    parser.add_argument("--geometry_dir", type=Path,
                        help="Saved crop geometry; defaults to data_dir/<subset>/geometry")
    parser.add_argument("--postprocessing_config", type=Path,
                        default=Path(__file__).with_name("postprocessing_config.json"))
    parser.add_argument("--reference_run", type=Path,
                        help="Run containing frozen anatomy and dense CRF configs")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", action="store_true")
    devices.add_argument("--mps", action="store_true")
    args = parser.parse_args()
    if not args.postprocess_only and args.data_dir is None:
        parser.error("Inference requires --data_dir")
    if args.postprocess_only or args.export_volumes or args.postprocess == "q9c":
        if args.num_classes != 5:
            parser.error("Native volume export and Q9c currently require five SegTHOR classes")
        if not args.source_scan_pattern or "{id_}" not in args.source_scan_pattern:
            parser.error("Volume export requires --source_scan_pattern containing {id_}")
        if args.geometry_dir is None:
            if args.data_dir is None:
                parser.error("--postprocess_only requires --geometry_dir or --data_dir")
            args.geometry_dir = args.data_dir / args.subset / "geometry"
        if not args.geometry_dir.is_dir():
            parser.error(f"Missing saved crop geometry: {args.geometry_dir}")
    return args


if __name__ == "__main__":
    run(get_args())
