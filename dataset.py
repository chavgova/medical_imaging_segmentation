#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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

import random
from pathlib import Path
from typing import Callable, Union

import numpy as np
from matplotlib import image
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


def _is_empty_gt(gt_path: Path) -> bool:
    return not np.asarray(Image.open(gt_path)).any()


def _drop_empty_gt_slices(files: list[tuple[Path, Path | None]],
                          drop_fraction: float) -> list[tuple[Path, Path | None]]:
    """Randomly drop a fraction of purely-background (empty) GT slices, with a fixed seed."""
    assert 0.0 <= drop_fraction <= 1.0, drop_fraction

    empty: list[tuple[Path, Path | None]] = []
    non_empty: list[tuple[Path, Path | None]] = []
    for pair in files:
        _, gt_path = pair
        (empty if _is_empty_gt(gt_path) else non_empty).append(pair)

    keep_n = round(len(empty) * (1 - drop_fraction))
    kept_empty = random.Random(0).sample(empty, keep_n)

    kept = non_empty + kept_empty
    kept.sort(key=lambda pair: pair[0])
    return kept


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
             gt_transform=None, augment=False, equalize=False, debug=False,
             context_slices: int = 0, drop_empty_slices: float = 0.0):

        self.root_dir: str = root_dir
        self.img_transform: Callable | None = img_transform
        self.gt_transform: Callable | None = gt_transform
        self.augmentation: bool = augment and subset == 'train'
        self.equalize: bool = equalize
        self.context_slices: int = context_slices
        assert self.context_slices >= 0

        self.test_mode: bool = subset == 'test'

        all_files = make_dataset(root_dir, subset)
        
        if drop_empty_slices and subset == 'train':
            all_files = _drop_empty_gt_slices(all_files, drop_empty_slices)

        self.context_slices: int = context_slices
        assert self.context_slices >= 0

        if self.context_slices > 0:
            self.imgs_by_sequence = {}

            for img_path, _ in all_files:
                prefix, width, z = self._parse_slice(img_path)
                key = (prefix, width)
                self.imgs_by_sequence.setdefault(key, {})[z] = img_path

            self.sequence_bounds = {
                key: (min(slices), max(slices))
                for key, slices in self.imgs_by_sequence.items()
            }

        self.files = all_files
        
        if debug:
            self.files = self.files[:10]
        
        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def slice_has_foreground(self, index: int) -> bool:
        """Checks whether a slice contains foreground (i.e. not only
        background). Used for oversampling images with foreground."""
        _, gt_path = self.files[index]
        if gt_path is None:
            return False
        return bool(np.asarray(Image.open(gt_path)).any())

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        gt_pil = None
        if not self.test_mode:
            with Image.open(gt_path) as mask:
                gt_pil = mask.copy()

        # We don't need a check on whether we do 2.5D
        # No 2.5D is just a special case with context_slices=0
        offsets = range(-self.context_slices, self.context_slices + 1)
        neighbor_imgs = []
        for offset in offsets:
            with Image.open(self._neighbor_img_path(img_path, offset)) as im:
                neighbor_imgs.append(self.img_transform(im.copy()))
        img: Tensor = torch.cat(neighbor_imgs, dim=0)

        if self.augmentation and torch.rand(()).item() < 0.5:
            angle = torch.empty(()).uniform_(-5.0, 5.0).item()
            img = TF.rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            gt_pil = TF.rotate(gt_pil, angle=angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)

        if self.augmentation and torch.rand(()).item() < 0.25:
            noise_std = torch.empty(()).uniform_(0.0, 0.01).item()
            img = (img + torch.randn_like(img) * noise_std).clamp(0.0, 1.0)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt_pil)

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt
        return data_dict
    
    def _parse_slice(self, img_path: Path) -> tuple[str, int, int]: 
        stem = img_path.stem  

        if "_" in stem:
            prefix, index = stem.rsplit("_", 1)
        else: 
            prefix, index = "", stem

        if not index.isdigit():
            raise ValueError( f"Cannot use 2.5D context for file with non-numeric slice suffix: {img_path}")
        return prefix, len(index), int(index)

    def _neighbor_img_path(self, img_path: Path, offset: int) -> Path:
        if offset == 0: 
            return img_path 

        prefix,width, z = self._parse_slice(img_path) 
        key = (prefix,width)

        sequence = self.imgs_by_sequence[key]
        min_z, max_z = self.sequence_bounds[key]

        requested_z = z + offset
        neighbor_z = min(max(requested_z, min_z), max_z)

        neighbor_path =  sequence.get(neighbor_z)
        if neighbor_path is None:   
            raise FileNotFoundError(  
                f"Missing neighboring slice {neighbor_z:0{width}d} for {img_path}"
                f"inside sequence range [{min_z}, {max_z}].")  

        return neighbor_path