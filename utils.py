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

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
import copy
import json

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum
from torch import nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode
import torch.nn.functional as F

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


# FLOPs profiling
FLOPS_PROFILE_BATCHES = 10 

def _profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU] 
    if device.type == "cuda": 
        activities.append(ProfilerActivity.CUDA)

    return activities  


def _profiler_flops(prof) -> int:
    total=0 
    for event in prof.key_averages():
        if event.flops is not None: 
            total += event.flops

    return int(total)


def _profile_train_batch(net: nn.Module,optimizer, loss_fn,img: Tensor, gt: Tensor, device: torch.device) -> int:
    net.train() 

    with profile(activities=_profiler_activities(device), with_flops=True) as prof:
        optimizer.zero_grad() 
        pred_logits = net(img) 
        pred_probs = F.softmax(pred_logits, dim=1)
        loss = loss_fn(pred_probs, gt)
        loss.backward()
        optimizer.step()

    return _profiler_flops(prof)


def _profile_train_batch_with_backprop(net: nn.Module, optimizer, loss_fn, img: Tensor, gt: Tensor, device: torch.device) -> int:
    """torch profiler doesn't include backpropagation, this does."""
    net.train()

    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        with FlopCounterMode(display=False) as counter:
            optimizer.zero_grad()
            pred_logits = net(img)
            pred_probs = F.softmax(pred_logits, dim=1)
            loss = loss_fn(pred_probs, gt)
            loss.backward()
            optimizer.step()

    return counter.get_total_flops()


def _profile_val_batch(net: nn.Module, img: Tensor, device: torch.device) -> int:
    net.eval()

    with profile(activities=_profiler_activities(device), with_flops=True) as prof:
        with torch.no_grad():
            pred_logits = net(img) 
            bla = F.softmax(pred_logits, dim=1) 

    return _profiler_flops(prof)


def estimate_flops(net: nn.Module, optimizer, loss_fn, train_loader, val_loader, device: torch.device) -> dict:
    initial_state = copy.deepcopy(net.state_dict())

    train_sample_flops: list[float] = []
    train_sample_flops_with_backprop: list[float] = []
    for i, data in enumerate(train_loader):
        if i >= FLOPS_PROFILE_BATCHES:
            break

        img = data["images"].to(device)
        gt = data["gts"].to(device)

        flops = _profile_train_batch(net, optimizer, loss_fn, img, gt, device)
        train_sample_flops.append(flops / img.shape[0])

        flops = _profile_train_batch_with_backprop(net, optimizer, loss_fn, img, gt, device)
        train_sample_flops_with_backprop.append(flops / img.shape[0])

    val_sample_flops: list[float] = []
    for i, data in enumerate(val_loader):
        if i >= FLOPS_PROFILE_BATCHES:
            break

        img = data["images"].to(device)

        flops = _profile_val_batch(net, img, device)
        val_sample_flops.append(flops / img.shape[0])

    net.load_state_dict(initial_state)
    del initial_state  
    optimizer.state.clear()

    return {
        "flops_per_train_sample": float(np.mean(train_sample_flops)) if train_sample_flops else None,
        "flops_per_train_sample_with_backprop": float(np.mean(train_sample_flops_with_backprop)) if train_sample_flops_with_backprop else None,
        "flops_per_val_sample": float(np.mean(val_sample_flops)) if val_sample_flops else None
    }


def save_flops_count(dest: Path, flops_count: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "flops_count.json", "w") as f:
        json.dump(flops_count, f, indent=2, sort_keys=True)
