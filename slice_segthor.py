#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


def norm_arr(img: np.ndarray, hu_min=None, hu_max=None, use_clahe: bool = False) -> np.ndarray:
    casted = img.astype(np.float32)
    if use_clahe:
        from skimage.exposure import equalize_adapthist

        # Clip to a broad physiological range first so outliers (metal, scanner table)
        # don't skew the local histogram in whichever tile they land in.
        clip_min, clip_max = -1000.0, 300.0
        rescaled = (np.clip(casted, clip_min, clip_max) - clip_min) / (clip_max - clip_min)

        # 2D per axial slice: the network only ever sees one slice at a time, so there's
        # no benefit to 3D-aware tiling, only extra compute.
        equalized = np.empty_like(rescaled)
        for idz in range(rescaled.shape[2]):
            equalized[:, :, idz] = equalize_adapthist(rescaled[:, :, idz], clip_limit=0.01)
        return (255 * equalized).astype(np.uint8)

    if hu_min is not None:
        import torchio as tio

        image = tio.ScalarImage(tensor=casted[None])
        image = tio.Clamp(out_min=hu_min, out_max=hu_max)(image)
        image = tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(hu_min, hu_max))(image)
        # Keep the existing PNG format; the loader divides by 255 during training.
        return (255 * image.data[0].numpy()).astype(np.uint8)

    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def center_crop_or_pad(arr: np.ndarray, shape: tuple[int, int], pad_value: int) -> np.ndarray:
    """Center-crop (if larger) or zero/background-pad (if smaller) arr to shape."""
    out = np.full(shape, pad_value, dtype=arr.dtype)

    src_h, src_w = arr.shape
    dst_h, dst_w = shape
    crop_h, crop_w = min(src_h, dst_h), min(src_w, dst_w)

    src_y0, src_x0 = (src_h - crop_h) // 2, (src_w - crop_w) // 2
    dst_y0, dst_x0 = (dst_h - crop_h) // 2, (dst_w - crop_w) // 2

    out[dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w] = \
        arr[src_y0:src_y0 + crop_h, src_x0:src_x0 + crop_w]
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def split_merged_aorta_esophagus(gt: np.ndarray, r: int = 4) -> np.ndarray:
    """ In the GT files aorta and esophagus are merged into the same label (1). This function takes the GT and splits them into two separate labels (1 for esophagus, 4 for aorta) using erotion and dilation operations. The aorta is much thicker
    than the esophagus, so eroding by r voxels leaves only the aorta's core.
    
    Small leftover islands on either side are reassigned to the other class, since
    both the real esophagus and the real aorta are each a single connected tube.
    Validated at 0.99 aorta Dice against Patient_07's known-correct split.
    """
    combined_mask = gt == 1
    if not combined_mask.any():
        return gt

    coords = np.argwhere(combined_mask)
    pad = 8
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, combined_mask.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    mask_c = combined_mask[sl]

    core = keep_largest_component(ndimage.binary_erosion(mask_c, iterations=r))
    aorta_c = ndimage.binary_dilation(core, iterations=r) & mask_c

    esophagus_c = mask_c & ~aorta_c
    aorta_c = aorta_c | (esophagus_c & ~keep_largest_component(esophagus_c))  # eso islands -> aorta
    aorta_c = keep_largest_component(aorta_c)  # aorta islands -> esophagus

    aorta = np.zeros_like(combined_mask)
    aorta[sl] = aorta_c

    corrected = gt.copy()
    corrected[aorta] = 4
    return corrected


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, hu_min=None, hu_max=None, use_clahe: bool = False,
                  target_spacing=None, fix_aorta_esophagus: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
        if fix_aorta_esophagus:
            gt = split_merged_aorta_esophagus(gt)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    norm_ct: np.ndarray = norm_arr(ct, hu_min, hu_max, use_clahe)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    # Baseline (target_spacing=None): resize straight to `shape`, ignoring native spacing,
    # as before. With target_spacing set: resize so 1 pixel = target_spacing mm for every
    # patient (dx == dy per sanity_ct), then center-crop/pad to the fixed `shape` for batching.
    if target_spacing is not None:
        scale = dx / target_spacing
        resize_shape = (max(1, round(x * scale)), max(1, round(y * scale)))
    else:
        resize_shape = shape

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], resize_shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], resize_shape, order=0).astype(np.uint8)
        if target_spacing is not None:
            img_slice = center_crop_or_pad(img_slice, shape, pad_value=0)
            gt_slice = center_crop_or_pad(gt_slice, shape, pad_value=0)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    if args.test_pipeline:
        ids = sorted(p.name for p in (src_path / 'train').glob('Patient_*')
                     if p.is_dir())
        training_ids, validation_ids = [ids[0]], [ids[1]]
    else:
        training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test',
                                 hu_min=args.hu_min,
                                 hu_max=args.hu_max,
                                 use_clahe=args.clahe,
                                 target_spacing=args.target_spacing,
                                 fix_aorta_esophagus=args.fix_aorta_esophagus)
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--hu_min', type=float, default=None, help='HU window lower bound; requires --hu_max.')
    parser.add_argument('--hu_max', type=float, default=None, help='HU window upper bound; requires --hu_min.')
    parser.add_argument('--clahe', action='store_true',
                        help='Clip to [-1000, 300] HU then apply CLAHE (2D per-slice) instead of '
                             'linear normalization. Mutually exclusive with --hu_min/--hu_max. '
                             'Default: off.')
    parser.add_argument('--target_spacing', type=float, default=None,
                        help='Resample in-plane to this physical spacing (mm/pixel) before '
                             'center-crop/pad to --shape. Default: no resampling (baseline), '
                             'native per-patient spacing is ignored like before.')
    parser.add_argument('--fix_aorta_esophagus', action='store_true',
                        help='Split the aorta back out of the merged esophagus label (1) using '
                             'erosion/dilation by shape. Default: off, keeps the original merged '
                             'label unchanged. See split_merged_aorta_esophagus() for details.')
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--test_pipeline', action='store_true',
                        help='Use the first two patients for a smoke run.')
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    if (args.hu_min is None) != (args.hu_max is None):
        parser.error('Supply --hu_min and --hu_max together')
    if args.hu_min is not None and not (np.isfinite(args.hu_min) and np.isfinite(args.hu_max)
                                        and args.hu_min < args.hu_max):
        parser.error('HU bounds must be finite, with --hu_min < --hu_max')
    if args.clahe and args.hu_min is not None:
        parser.error('--clahe and --hu_min/--hu_max are mutually exclusive normalization choices')
    if args.target_spacing is not None and not (np.isfinite(args.target_spacing)
                                                 and args.target_spacing > 0):
        parser.error('--target_spacing must be a finite positive number')
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
