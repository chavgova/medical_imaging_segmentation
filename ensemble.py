#!/usr/bin/env python3
"""Ensemble inference from a folder of full PyTorch models.

By default, find *.pkl models recursively (including training's bestmodel.pkl).
For other filenames, pass --model_pattern, e.g. '**/*.pt'. These must be
trusted full models saved with torch.save(model, path), not state dictionaries.
Members must share preprocessing, input channels and output class ordering.
Uses main.py's SliceDataset and transforms; train/val subsets require ground
truth PNGs, while the test subset only requires input images.

Outputs:
  predictions/<subset>/<stem>.png: same label encoding as main.py.
  uncertainty/<subset>/<stem>.npz: normalized entropy (H, W), normalized
    mutual_information (H, W), and population variance per class (K, H, W).
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

from utils import save_images, tqdm_



def load_models(folder: Path, pattern: str, device: torch.device,
                context_slices: int, class_count: int):
    if not folder.is_dir():
        raise ValueError(f"Ensemble folder does not exist: {folder}")
    paths = sorted(path for path in folder.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No models matching {pattern!r} in {folder}")
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
    from main import SliceDataset, img_transform, gt_transform

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
                               args.context_slices, args.num_classes)
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
    print(f">> Ensemble of {len(models)} models on {device}: {[str(path) for path in paths]}")
    for batch in tqdm_(loader):
        mean, maps = predict_ensemble(models, batch["images"].to(device), args.num_classes)
        multiplier = 63 if args.num_classes == 5 else 255 / (args.num_classes - 1)
        save_images(mean.argmax(dim=1) * multiplier, batch["stems"], prediction_dir)
        maps = {key: value.cpu().numpy() for key, value in maps.items()}
        for index, stem in enumerate(batch["stems"]):
            np.savez_compressed(uncertainty_dir / f"{stem}.npz",
                                **{key: value[index] for key, value in maps.items()})
    print(f">> Predictions: {prediction_dir}; uncertainty: {uncertainty_dir}")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble_dir", "--ensemble", type=Path, required=True)
    parser.add_argument("--model_pattern", default="**/*.pkl")
    parser.add_argument("--data_dir", type=Path, required=True, help="Processed data root")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--uncertainty_dest", type=Path)
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--context_slices", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", action="store_true")
    devices.add_argument("--mps", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
