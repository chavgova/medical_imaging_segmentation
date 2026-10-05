"""Post-processing functions for 3D numpy segmentation label maps."""

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import ndimage
import pydensecrf.densecrf as dcrf

def _validate_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("Expected a 3D label map.")
    if not (np.issubdtype(volume.dtype, np.integer) or volume.dtype == np.bool_):
        if not (np.issubdtype(volume.dtype, np.floating)
                and np.isfinite(volume).all()
                and np.equal(volume, np.floor(volume)).all()):
            raise ValueError("Expected finite integer-valued labels.")
    return volume


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def keep_largest_connected_components(
    volume: np.ndarray,
    k: int = 1,
    classes: Optional[Sequence[int]] = None,
    connectivity: int = 26,
) -> np.ndarray:
    """Keep up to k largest 3D components independently for each selected class.

    Components are ranked by voxel count. Connectivity is 6 (faces), 18
    (faces and edges), or 26 (faces, edges and corners). Equal-size ties
    favor the component encountered first in array traversal order.

    By default, process all nonzero labels. Label 0 is background and is
    never filtered. Unselected classes are unchanged; discarded voxels
    become background. Returns a copy with the original shape and dtype.
    This is size-based filtering, not a test of anatomical plausibility.
    """
    volume = _validate_volume(volume)
    _validate_positive_integer(k, "k")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26.")

    selected = np.unique(volume) if classes is None else classes
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    result = volume.copy()
    for c in selected:
        if c == 0:
            continue
        mask = volume == c
        components, count = ndimage.label(mask, structure=structure)
        if count <= k:
            continue
        sizes = np.bincount(components.ravel())[1:]
        largest = np.argsort(-sizes, kind="stable")[:k] + 1
        keep = np.zeros(count + 1, dtype=bool)
        keep[largest] = True
        result[mask & ~keep[components]] = 0
    return result


def keep_foreground_components(volume: np.ndarray, k: int) -> np.ndarray:
    """Connect all foreground labels together, but preserve surviving labels."""
    volume = _validate_volume(volume)
    keep = keep_largest_connected_components(volume != 0, k=k, connectivity=26)
    result = volume.copy()
    result[~keep] = 0
    return result


def remove_small_components(volume: np.ndarray, min_size: int) -> np.ndarray:
    """Remove per-class components with fewer than min_size voxels."""
    volume = _validate_volume(volume)
    _validate_positive_integer(min_size, "min_size")
    result = volume.copy()
    structure = ndimage.generate_binary_structure(3, 3)
    for c in np.unique(volume):
        if c == 0:
            continue
        mask = volume == c
        components, _ = ndimage.label(mask, structure)
        sizes = np.bincount(components.ravel())
        keep = sizes >= min_size
        keep[0] = False
        result[mask & ~keep[components]] = 0
    return result


def anatomy_aware_filtering(
    volume: np.ndarray,
    spacing: Sequence[float],
    config: Mapping[str, Any],
) -> np.ndarray:
    """Filter 3D components using class-specific physical-size rules.

    Components are processed independently per configured foreground class.
    A component must satisfy both the minimum physical volume and minimum
    fraction of its class's largest component. The number retained can also be
    capped per class. This function only removes voxels
    """
    volume = _validate_volume(volume)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (3,) or not np.isfinite(spacing_array).all() or np.any(spacing_array <= 0):
        raise ValueError("spacing must contain 3 finite positive values")

    connectivity = config.get("connectivity", 26)
    
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26.")
    class_rules = config.get("classes")

    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    voxel_volume_mm3 = float(np.prod(spacing_array))
    result = volume.copy()

    allowed_rule = {
        "name",
        "min_volume_mm3",  #min_volume_mm3 must be finite and non-negative
        "min_relative_volume", #  min_relative_volume must be in [0, 1]
        "max_components",
        "always_keep_largest", # boolean 
    }
    for class_key, raw_rule in class_rules.items():
        class_id = int(class_key)

        min_volume_mm3 = raw_rule.get("min_volume_mm3", 0.0)
        min_relative_volume = raw_rule.get("min_relative_volume", 0.0)
        max_components = raw_rule.get("max_components")
        always_keep_largest = raw_rule.get("always_keep_largest", True)

        mask = volume == class_id
        components, count = ndimage.label(mask, structure=structure)
        if count == 0:
            continue

        voxel_counts = np.bincount(components.ravel())[1:]
        component_volumes = voxel_counts.astype(np.float64) * voxel_volume_mm3
        largest_volume = float(component_volumes.max())
        keep_components = (
            (component_volumes >= float(min_volume_mm3))
            & (component_volumes >= float(min_relative_volume) * largest_volume)
        )
        largest_index = int(np.argmax(component_volumes))
        if always_keep_largest:
            keep_components[largest_index] = True

        retained = np.flatnonzero(keep_components)
        if max_components is not None and retained.size > max_components:
            order = np.argsort(-component_volumes[retained], kind="stable")
            retained = retained[order[:max_components]]

        keep_lookup = np.zeros(count + 1, dtype=bool)
        keep_lookup[retained + 1] = True
        result[mask & ~keep_lookup[components]] = 0

    return result


def dense_crf(volume: np.ndarray, probabilities: np.ndarray, image: np.ndarray, spacing: Sequence[float], config: Mapping[str, Any]) -> np.ndarray:
    """Refines a multiclass 3D label map with CT-aware dense CRF inference"""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    spacing_array = np.asarray(spacing, dtype=np.float32)

    foreground = probabilities[..., 1:].sum(axis=-1) >= config["roi_foreground_probability"]
    foreground_voxels = np.argwhere(foreground)
    if foreground_voxels.size == 0:
        return volume.copy()

    margin = np.ceil(config["roi_margin_mm"] / spacing_array).astype(int)
    start =np.maximum(foreground_voxels.min(axis=0) - margin,0)
    stop = np.minimum(foreground_voxels.max(axis=0)+ margin + 1,volume.shape)
    roi = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))

    roi_probabilities = np.moveaxis(probabilities[roi], -1, 0)
    epsilon = config["probability_epsilon"]
    roi_probabilities = np.clip(roi_probabilities, epsilon,1.0)
    roi_probabilities /= roi_probabilities.sum(axis=0, keepdims=True)  

    class_count, *roi_shape = roi_probabilities.shape  
    voxel_count = int(np.prod(roi_shape))  
    crf = dcrf.DenseCRF(voxel_count, class_count) 
    unary = -np.log(roi_probabilities).reshape(class_count, -1)   
    crf.setUnaryEnergy(np.ascontiguousarray(unary, dtype=np.float32))
 
    coordinates = np.indices(roi_shape, dtype=np.float32).reshape(3, -1)
    coordinates *= spacing_array[:, None]  
 
    spatial_sigma = np.asarray(config["spatial_sigma_mm"], dtype=np.float32)[:, None]
    spatial_features = np.ascontiguousarray(coordinates / spatial_sigma, dtype=np.float32)
    crf.addPairwiseEnergy(spatial_features, compat=config["spatial_weight"], kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    bilateral_sigma = np.asarray(config["bilateral_spatial_sigma_mm"], dtype=np.float32)[:, None]
    intensity = image[roi].reshape(1, -1) / config["bilateral_intensity_sigma_hu"]
    bilateral_features = np.vstack((coordinates / bilateral_sigma, intensity))
    crf.addPairwiseEnergy( 
        np.ascontiguousarray(bilateral_features, dtype=np.float32),
        compat=config["bilateral_weight"], 
        kernel=dcrf.DIAG_KERNEL, 
        normalization=dcrf.NORMALIZE_SYMMETRIC)
 
    posterior = np.asarray(crf.inference(config["iterations"]), dtype=np.float32)
    result = volume.copy()
    result[roi] = posterior.argmax(axis=0).reshape(roi_shape).astype(volume.dtype)
    return result 


def fill_holes(
    volume: np.ndarray,
    classes: Optional[Sequence[int]] = None,
    connectivity: int = 6,
) -> np.ndarray:
    """Fill enclosed 3D cavities per class using scipy.ndimage.binary_fill_holes.

    Connectivity (6, 18, or 26) describes the background paths to the volume
    boundary. Only original background voxels can be filled; existing class
    labels are preserved. If multiple selected classes claim the same cavity
    voxel (e.g. nested shells), leave it as background rather than choosing a
    class by iteration order. Returns a copy of the input shape and dtype.
    """
    volume = _validate_volume(volume)
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26.")
    selected = np.unique(volume) if classes is None else np.unique(classes)
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    background = volume == 0
    claimed = np.zeros(volume.shape, dtype=bool)
    result = volume.copy()
    for c in selected:
        if c == 0:
            continue
        mask = volume == c
        if not mask.any():
            continue
        candidates = ndimage.binary_fill_holes(mask, structure=structure) & background
        result[candidates & ~claimed] = c
        result[candidates & claimed] = 0
        claimed |= candidates
    return result


def _morphological_cleanup(volume, operation, iterations, classes, connectivity):
    volume = _validate_volume(volume)
    _validate_positive_integer(iterations, "iterations")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26.")
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    selected = np.unique(volume) if classes is None else np.unique(classes)
    result = volume.copy()
    background = volume == 0
    claimed = np.zeros(volume.shape, dtype=bool) if operation == "closing" else None
    for c in selected:
        if c == 0:
            continue
        mask = volume == c
        if not mask.any():
            continue
        if operation == "opening":
            opened = ndimage.binary_opening(mask, structure=structure, iterations=iterations)
            result[mask & ~opened] = 0
        else:
            # Allow dilation outside the image before erosion, so closing does
            # not accidentally erase foreground touching the image boundary.
            padded = np.pad(mask, iterations, mode="constant")
            closed = ndimage.binary_closing(padded, structure=structure, iterations=iterations)
            crop = (slice(iterations, -iterations),) * 3
            candidates = closed[crop] & background
            result[candidates & ~claimed] = c
            result[candidates & claimed] = 0
            claimed |= candidates
    return result


def opening(
    volume: np.ndarray,
    iterations: int = 1,
    classes: Optional[Sequence[int]] = None,
    connectivity: int = 6,
) -> np.ndarray:
    """Per-class 3D binary erosion followed by dilation, using SciPy.

    Apply `iterations` erosions then the same number of dilations with a
    6-, 18-, or 26-neighbor structuring element. Only remove original voxels
    of selected classes; preserve all other labels. Outside the image is
    background. Returns a copy with the original shape and dtype.
    """
    return _morphological_cleanup(volume, "opening", iterations, classes, connectivity)


def closing(
    volume: np.ndarray,
    iterations: int = 1,
    classes: Optional[Sequence[int]] = None,
    connectivity: int = 6,
) -> np.ndarray:
    """Per-class 3D binary dilation followed by erosion, using SciPy.

    Apply `iterations` dilations then the same number of erosions. Pad with
    background before the operation to preserve boundary-touching labels.
    Only fill original background voxels. Preserve existing labels and leave
    competing class claims as background. Returns a copy of the input.
    """
    return _morphological_cleanup(volume, "closing", iterations, classes, connectivity)


def salt_and_pepper(
    volume: np.ndarray,
    kernel_size: int = 3,
    classes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Denoise each selected class with a 3D binary median filter.

    Use an odd cubic window (default 3x3x3) with nearest-edge padding. Remove
    class voxels rejected by its binary median and fill original background
    voxels accepted by it. Never directly replace one foreground label with
    another. All masks come from the original volume, so class order has no
    effect. An odd-window binary median requires a strict majority; two
    different classes cannot both claim the same background voxel.

    Returns a copy with the original shape and dtype. kernel_size=1 is a no-op.
    """
    volume = _validate_volume(volume)
    _validate_positive_integer(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    result = volume.copy()
    if kernel_size == 1:
        return result
    selected = np.unique(volume) if classes is None else np.unique(classes)
    background = volume == 0
    for c in selected:
        if c == 0:
            continue
        mask = volume == c
        if not mask.any():
            continue
        filtered = ndimage.median_filter(mask, size=kernel_size, mode="nearest")
        result[mask & ~filtered] = 0
        result[background & filtered] = c
    return result
