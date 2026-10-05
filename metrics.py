#!/usr/bin/env python3

# Dataset-independent segmentation evaluation metrics: IoU, Dice, HD95, ASD,
# and a voxel-wise confusion matrix.

# Every function takes plain numpy label maps (not one-hot, not torch tensors)
# and a class value to score (confusion_matrix covers all classes at once), so
# they work the same way for a 2-class (binary) problem or an N-class one, on
# any dataset.
# HD95 and ASD are computed by MONAI (monai.metrics); the wrappers here only
# convert the inputs and apply our empty-mask conventions.

from typing import Optional, Sequence

import numpy as np
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance

def iou(pred: np.ndarray, gt: np.ndarray, c: int = 1, eta: float = 1e-8) -> float:
    """
    Parameters
    ----------
    pred, gt:
        Integer (or boolean) label maps of identical shape. Any
        dimensionality is supported (2D slice or 3D volume)
    c:
        The class value to score. For an already-boolean mask, the default
        c=1 selects the foreground.
    """
    assert pred.shape == gt.shape, (pred.shape, gt.shape)

    pred_mask = pred == c
    gt_mask = gt == c

    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64) + eta
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.int64) + eta

    return float(intersection / union)

def dice(pred: np.ndarray, gt: np.ndarray, classes: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Parameters
    ----------
    pred, gt:
        Integer label maps of identical shape.
    classes:
        The class values to score.

    Returns
    -------
    np.ndarray
        1D array of shape (len(classes),), one Dice score per class in
        [0, 1]. A class absent from both pred and gt yields Dice = 1.0,
    """
    if classes is None:
        classes = sorted(set(np.unique(pred).tolist()) | set(np.unique(gt).tolist()))

    ious = np.array([iou(pred, gt, c) for c in classes], dtype=np.float64)

    return 2 * ious / (1 + ious)

def confusion_matrix(gt: np.ndarray, pred: np.ndarray, n_classes: int = 5,
                     spacing: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Voxel-wise confusion matrix: rows are ground-truth labels, columns are
    predicted labels, so entry [i, j] is the number of voxels of true class i
    predicted as class j.

    Parameters
    ----------
    gt, pred:
        Integer label maps of identical shape, with labels in [0, n_classes).
        Note the (gt, pred) order, unlike the (pred, gt) of the other metrics.
    n_classes:
        Number of classes including background (5 for SegTHOR).
    spacing:
        Optional physical voxel size (e.g. (sx, sy, sz) in mm), one value per
        axis of gt/pred.

    Returns
    -------
    np.ndarray
        (n_classes, n_classes) matrix of int64 voxel counts, or of float64
        volumes in mm3 (counts x voxel volume) when spacing is given.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred must have the same shape, got {gt.shape} and {pred.shape}")

    gt = np.asarray(gt).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    for name, labels in (("gt", gt), ("pred", pred)):
        if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
            raise ValueError(f"{name} labels must be in [0, {n_classes - 1}], "
                             f"got values from {labels.min()} to {labels.max()}")

    # each (gt, pred) pair maps to a unique bin n_classes * gt + pred
    counts = np.bincount((n_classes * gt + pred).ravel(), minlength=n_classes ** 2)
    counts = counts.reshape(n_classes, n_classes)

    if spacing is None:
        return counts

    if len(spacing) != gt.ndim:
        raise ValueError(f"spacing needs one value per axis ({gt.ndim}), got {tuple(spacing)}")

    return counts * float(np.prod(spacing))

def _to_monai(mask: np.ndarray, device: str) -> torch.Tensor:
    """Binary mask (X, Y, Z) -> MONAI's batch-first one-hot layout (1, 1, X, Y, Z), on `device`."""
    return torch.from_numpy(mask)[None, None].to(device)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], c: int = 1, device: str = "cpu") -> float:
    """

    Parameters
    ----------
    pred, gt:
        3D integer (or boolean) label maps of identical shape, `(X, Y, Z)`.
    spacing:
        Physical voxel size (sx, sy, sz) in mm, matching the axis order of
        pred/gt.
    c:
        The class value to score.
    device:
        Where MONAI computes the distances. On "cuda" it uses cuCIM's GPU
        erosion and distance transform, so cuCIM and CuPy must be installed
        (without them MONAI falls back to SciPy on CPU, slower than "cpu").

    Returns
    -------
    float
        HD95 in mm: `max(P95(pred->gt distances), P95(gt->pred distances))`,
        computed by `monai.metrics.compute_hausdorff_distance`.
    """
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred.ndim == len(spacing), (pred.shape, spacing)

    pred_mask = pred == c
    gt_mask = gt == c

    # no boundary disagreement is possible
    if not pred_mask.any() and not gt_mask.any():
        return 0.0

    #  no reference surface on the empty side
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")

    # the single channel is the class being scored, so it must not be dropped as background
    hd95 = compute_hausdorff_distance(_to_monai(pred_mask, device), _to_monai(gt_mask, device), include_background=True, percentile=95, directed=False, spacing=tuple(map(float, spacing)))

    return float(hd95[0, 0])

def average_surface_distance(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], c: int = 1, device: str = "cpu") -> float:
    """

    Parameters
    ----------
    pred, gt:
        3D label maps of identical shape (X, Y, Z).
    spacing:
        Physical voxel size (sx, sy, sz) in mm, matching the axis order of
        pred/gt
    c:
        The class value to score.
    device:
        Where MONAI computes the distances. On "cuda" it uses cuCIM's GPU
        erosion and distance transform, so cuCIM and CuPy must be installed
        (without them MONAI falls back to SciPy on CPU, slower than "cpu").

    Returns
    -------
    float
        The mean, in mm, of the pooled pred->gt and gt->pred surface
        distances (one mean over every surface voxel on both sides),
        computed by `monai.metrics.compute_average_surface_distance`.
    """
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred.ndim == len(spacing), (pred.shape, spacing)

    pred_mask = pred == c
    gt_mask = gt == c

    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return float("nan")

    # symmetric=True: MONAI defaults to the directed pred->gt distance only
    asd = compute_average_surface_distance(_to_monai(pred_mask, device), _to_monai(gt_mask, device), include_background=True, symmetric=True, spacing=tuple(map(float, spacing)))

    return float(asd[0, 0])

