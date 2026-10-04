#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import subprocess
import sys
import warnings
from typing import Any, Literal, get_args
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from UNet import UNet, FoundationFusion
from dino import Dino, FrozenDino
from slice_segthor import parse_hu_windows
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   estimate_flops,
                   save_flops_count,
                   load_preprocessing)

from losses import (CrossEntropy, Dice, DiceCE, Balance)
import json
import random

Model = Literal["shallowcnn", "enet", "unet-small", "unet-medium", "unet-large"]

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'B': 2, 'model': 'shallowcnn'}
datasets_params["SEGTHOR"] = {'K': 5, 'B': 8, 'model': 'enet'}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'B': 8, 'model': 'enet'}


def make_net(model: Model, in_channels: int, K: int, foundation_model: Dino | None = None,
             foundation_channels: int | None = None,
             foundation_fusion: FoundationFusion | None = None,
             hu_window: tuple[float, float] | None = None, context_slices: int = 0,
             n_windows: int = 1) -> shallowCNN | ENet | UNet:
    assert foundation_model is None or model.startswith("unet"), \
        f"--foundation_model only works with the U-Nets, not {model}"
    assert (foundation_model is None) == (foundation_fusion is None), \
        "--foundation_fusion must be given exactly when there is a --foundation_model"
    assert foundation_model is not None or foundation_channels is None, \
        "--foundation_channels is only used with a --foundation_model"
    frozen = FrozenDino(foundation_model, hu_window, context_slices, n_windows) \
        if foundation_model is not None else None
    match model:
        case "shallowcnn":
            return shallowCNN(in_channels, K)
        case "enet":
            return ENet(in_channels, K, kernels=8, factor=2)
        case "unet-small":
            return UNet(in_channels, K, kernels=8, max_channels=64, depth=6, norm="instance",
                        activation="leaky_relu", negative_slope=0.01, downsample="strided",
                        foundation_model=frozen, foundation_channels=foundation_channels,
                        foundation_fusion=foundation_fusion)
        case "unet-medium":
            return UNet(in_channels, K, kernels=16, max_channels=256, depth=6, norm="instance",
                        activation="leaky_relu", negative_slope=0.01, downsample="strided",
                        foundation_model=frozen, foundation_channels=foundation_channels,
                        foundation_fusion=foundation_fusion)
        case "unet-large":
            return UNet(in_channels, K, kernels=32, max_channels=512, depth=6, norm="instance",
                        activation="leaky_relu", negative_slope=0.01, downsample="strided",
                        foundation_model=frozen, foundation_channels=foundation_channels,
                        foundation_fusion=foundation_fusion)

def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)    
    
def make_scheduler(args: argparse.Namespace, optimizer):
    if args.scheduler == "none":
        return None

    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, 
                                               gamma=args.scheduler_gamma)

    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="max", 
                factor=args.scheduler_gamma, patience=args.scheduler_patience)

    if args.scheduler == "cosine":
        t_max = args.scheduler_t_max if args.scheduler_t_max > 0 else args.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    raise ValueError(f"Invalid scheduler {args.scheduler}")
    
def compute_invfreq_weights(loader: DataLoader, K: int, idk: list[int], alpha: float = 1.0) -> list[float]:
    """These are the inverse frequency weights as given by
    Sugino et al. (2021) in eq. 3.

        w_k = (N / (K * n_k)) ** alpha
    where 
        N = total pixel count, K = number of classes,
        n_k = pixel count of class k.

    alpha is the same power parameter), but with an added
    constant factor (counts.sum() / K) ** alpha, and the 
    requirement alpha > 0, so it can be interpreted that: 

    w = 1 for a class with an average number of pixels.
    w > 1 means the class is rarer (organs).
    w < 1 means the class is more common (background). 
    
    The constant factor doesn't matter for the loss, as the
    cross-entropy we use is normalized.

    Sugino et al. (2021): ./papers/healthcare-09-00938.pdf"""

    assert alpha > 0, f"alpha must be > 0 to keep rarer classes at w > 1, got {alpha}"

    counts = torch.zeros(K)
    for data in loader:
        counts += data["gts"].sum(dim=(0, 2, 3))
    weights = (counts.sum() / (K * (counts + 1e-8))) ** alpha
    return [weights[k].item() for k in idk]

def compute_foreground_oversampling_weights(dataset: SliceDataset, target_fg_fraction: float) -> list[float]:
    """To mitigate the problem that some images only contain background, we
    oversample the images that do contain foreground. This computes the
    per-sample weights needed for that:

        r = p * n_bg / ((1 - p) * n_fg)

    where p is the target fraction of samples per epoch that should contain
    foreground, and n_bg/n_fg are the background-only/foreground-containing
    sample counts."""
    has_fg = [dataset.slice_has_foreground(i) for i in range(len(dataset))]
    n_fg = sum(has_fg)
    n_bg = len(has_fg) - n_fg
    if n_fg == 0 or n_bg == 0:
        return [1.0] * len(has_fg)

    r = target_fg_fraction * n_bg / ((1 - target_fg_fraction) * n_fg)
    return [r if fg else 1.0 for fg in has_fg]

def save_config(args: argparse.Namespace) -> None:
    args.dest.mkdir(parents=True, exist_ok=True)

    payload = {key: str(value) if isinstance(value,Path) else value
        for key, value in vars(args).items()}

    with open(args.dest / "config.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    
# Data sliced before preprocessing.json existed: the exp_P1_HU window it was made with,
# so MedDINOv3 runs on that data keep working as before.
LEGACY_HU_WINDOW = (-1000.0, 300.0)


def read_preprocessing(args: argparse.Namespace, root_dir: Path) -> tuple[int, tuple[float, float] | None]:
    """The number of HU windows (image folders) in the data, and the HU window of the
    first one (for MedDINOv3), from the preprocessing.json written by slice_segthor.py."""
    preprocessing = load_preprocessing(root_dir)
    args.preprocessing = preprocessing  # also saved in config.json
    if preprocessing is None:
        return 1, LEGACY_HU_WINDOW

    windows = preprocessing["windows"]
    n_windows = len(windows) if preprocessing["method"] == "multiwindow" else 1
    # min-max and CLAHE don't keep a linear HU mapping
    hu_window = tuple(windows[0]) if preprocessing["method"] in ("hu", "multiwindow") else None
    return n_windows, hu_window


def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler + adding mps support
    gpu: bool = args.gpu and torch.cuda.is_available()
    mps: bool = args.mps and torch.backends.mps.is_available()
    device = torch.device("cuda") if gpu else torch.device("mps") if mps else torch.device("cpu")

    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    root_dir = args.data_dir if args.data_dir is not None else Path("data") / args.dataset
    n_windows, hu_window = read_preprocessing(args, root_dir)
    in_channels = (2 * args.context_slices + 1) * n_windows
    net = make_net(args.model, in_channels, K, args.foundation_model, args.foundation_channels,
                   args.foundation_fusion, hu_window, args.context_slices, n_windows)
    net.init_weights()
    net.to(device)

    lr = args.lr # default is  lr = 0.0005
        
    if args.opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Invalid optimizer {args.opt}")

    # Dataset part
    B: int = datasets_params[args.dataset]['B']

    generator = None
    worker_init_fn = None
    if args.deterministic:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        worker_init_fn = seed_worker

    train_set = SliceDataset(
        'train',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        augment=args.augment,
        debug=args.debug,
        drop_empty_slices=args.drop_empty_slices,
        context_slices=args.context_slices,
        n_windows=n_windows)

    train_sampler = None
    if args.oversample_foreground:
        fg_weights = compute_foreground_oversampling_weights(train_set, args.oversample_foreground_percent)
        train_sampler = WeightedRandomSampler(fg_weights, num_samples=len(train_set),
                                              replacement=True, generator=generator)

    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              worker_init_fn=worker_init_fn,
                              generator=generator)

    val_set = SliceDataset(
        'val',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
        context_slices=args.context_slices,
        n_windows=n_windows)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            generator=generator)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    save_config(args)
    scheduler = make_scheduler(args, optimizer)
    epochs_without_improvement = 0

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    ce_weights: list[float] | None = None
    if args.ce_weights == "invfreq":
        ce_weights = compute_invfreq_weights(train_loader, K, idk, alpha=args.ce_weights_alpha)
        print(f">> Computed invfreq CE weights: {ce_weights}")
    elif args.ce_weights is not None:
        ce_weights = [float(w) for w in args.ce_weights.split(",")]

    if ce_weights is not None and args.loss_fn not in ("ce", "dicece"):
        print(f">> Warning: --ce_weights has no effect for --loss_fn {args.loss_fn}")

    if args.loss_fn == "ce":
        loss_fn = CrossEntropy(idk=idk, weight=ce_weights)
    elif args.loss_fn == "dice":
        loss_fn = Dice(idk=idk)
    elif args.loss_fn == "dicece":
        loss_fn = DiceCE(idk=idk, lambda_=args.dicece_lambda, weight=ce_weights)
    elif args.loss_fn == "balance":
        if args.balance_fallback_epoch == -1:
            fallback_epoch = max(1, args.epochs // 2)
        elif args.balance_fallback_epoch == 0:
            fallback_epoch = None
        else:
            fallback_epoch = args.balance_fallback_epoch
        loss_fn = Balance(idk=idk, alpha=args.balance_alpha, t=args.balance_t,
                          normalized=args.balance_normalized, fallback_epoch=fallback_epoch,
                          inter=args.balance_inter)
    else:
        raise ValueError(f"Invalid loss function {args.loss_fn}")

    flops_count = estimate_flops(net, optimizer, loss_fn, train_loader, val_loader, device)  # pytorch FLOPs are estimates and mainly cover conv/matmul operations
    save_flops_count(args.dest, flops_count)

    if args.only_count_flops:
        return

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0
    epochs_ran: int = 0

    for e in range(args.epochs):
        epochs_ran = e + 1
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    if isinstance(loss_fn, Balance) and m == 'train':
                        loss_fn.step(pred_probs, gt)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        if isinstance(loss_fn, Balance):
            loss_fn.on_epoch_end(e)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        improved = current_dice > best_dice + args.early_stopping_min_delta

        if improved:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            epochs_without_improvement = 0

            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(current_dice)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f">>> Learning rate after epoch {e}: {current_lr:.3e}")

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f">>> Early stopping after {epochs_without_improvement} epochs without improvement")
            break

    args.epochs_ran = epochs_ran
    args.train_samples = len(train_loader.dataset)
    args.val_samples = len(val_loader.dataset)
    save_config(args)


def ensure_smoke_data(data_dir: Path, source_dir: Path, hu_min=None, hu_max=None, use_clahe=False,
                      target_spacing=None, fix_aorta_esophagus=False, hu_windows=None):
    """Create smoke data only if its directory does not exist."""
    if data_dir.exists():
        print(f'Reusing smoke dataset: {data_dir}')
        return

    print(f'Creating smoke dataset: {data_dir}', flush=True)
    command = [sys.executable, str(Path(__file__).with_name('slice_segthor.py')),
               '--source_dir', str(source_dir), '--dest_dir', str(data_dir), '--test_pipeline']
    if hu_min is not None:
        command += ['--hu_min', str(hu_min), '--hu_max', str(hu_max)]
    if use_clahe:
        command += ['--clahe']
    if hu_windows is not None:
        command += ['--hu_windows'] + [str(bound) for window in hu_windows for bound in window]
    if target_spacing is not None:
        command += ['--target_spacing', str(target_spacing)]
    if fix_aorta_esophagus:
        command += ['--fix_aorta_esophagus']
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--test_pipeline', action='store_true',
                        help='Run one epoch; for SEGTHOR, create or reuse the two-patient smoke dataset.')
    parser.add_argument('--dataset', default=None, choices=datasets_params.keys(),
                        help='Defaults to SEGTHOR with --test_pipeline, otherwise TOY2.')
    parser.add_argument('--model', default=None, choices=get_args(Model),
                        help="Defaults to the dataset's model (shallowcnn for TOY2, enet for SEGTHOR).")
    parser.add_argument('--foundation_model', default=None, choices=get_args(Dino),
                        help="Frozen foundation model whose patch features are concatenated into the "
                             "U-Net, at the level with the same resolution (16x16).")
    parser.add_argument('--foundation_channels', default=None, type=int,
                        help="Since foundation model features can be big we might want to "
                             "shrink them with a 1x1 conv before concatenating them into the U-Net. "
                             "Defaults to half of the U-Net's channels at that level.")
    parser.add_argument('--foundation_fusion', default=None, choices=get_args(FoundationFusion),
                        help="There are two places where the resolution of the U-Net feature are "
                        "the same as the foundation model's patch features: once in the encoder, once in the decoder. "
                        "I think the decoder is better since the DINO features are already contextualized. "
                        "But we could try both.")
    parser.add_argument('--data_dir', type=Path, default=None,
                        help='Processed train/val directory; defaults to data/SEGTHOR_smoke for '
                             'a SEGTHOR smoke run, otherwise data/<dataset>.')
    parser.add_argument('--source_dir', type=Path, default=Path('data/segthor_part1'),
                        help='Raw SegTHOR root containing train/, used for smoke preprocessing.')
    parser.add_argument('--hu_min', type=float, default=None,
                        help='Smoke preprocessing HU lower bound; requires --hu_max and a fresh --data_dir.')
    parser.add_argument('--hu_max', type=float, default=None,
                        help='Smoke preprocessing HU upper bound; requires --hu_min.')
    parser.add_argument('--clahe', action='store_true',
                        help='Smoke preprocessing: clip to the --hu_min/--hu_max window (default '
                             '[-1000, 300]) then CLAHE instead of linear normalization. '
                             'Default off; requires a fresh --data_dir.')
    parser.add_argument('--hu_windows', type=float, nargs='+', default=None, metavar='LO HI',
                        help='Smoke preprocessing: several HU windows as LO HI pairs, each its own '
                             'input channel. Mutually exclusive with --hu_min/--hu_max and --clahe; '
                             'requires a fresh --data_dir.')
    parser.add_argument('--target_spacing', type=float, default=None,
                        help='Smoke preprocessing target in-plane spacing (mm/pixel); '
                             'requires a fresh --data_dir, same as --hu_min/--hu_max.')
    parser.add_argument('--fix_aorta_esophagus', action='store_true',
                        help='Smoke preprocessing: split the merged aorta/esophagus label. '
                             'Default off; requires a fresh --data_dir, same as --hu_min/--hu_max.')
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path,
                        help='Results directory; required normally, defaults to '
                             'results/<dataset>/smoke_run with --test_pipeline.')

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--loss_fn', choices=["ce", "dice", "dicece", "balance"], default="ce", help="Loss function used during training.")
    parser.add_argument('--dicece_lambda', default=0.5, type=float,
                        help="Weight of the dice term when --loss_fn is dicece, in [0, 1]: "
                             "L = (1 - dicece_lambda) * L_CE + dicece_lambda * L_DICE.")
    parser.add_argument('--ce_weights', default=None, type=str,
                        help="Per-class weights for the CE term (applies to --loss_fn ce and dicece). Either "
                             "comma-separated floats matching the supervised classes, e.g. '0.5,1,1,2,3', or "
                             "'invfreq' to compute inverse-frequency weights from the training set.")
    parser.add_argument('--ce_weights_alpha', default=1.0, type=float,
                        help="Power parameter for --ce_weights invfreq (Sugino et al. 2021, eq. 3)")
    parser.add_argument('--balance_alpha', default=0.5, type=float,
                        help="Weight of the Intra-CBL term when --loss_fn is balance, in [0, 1]: "
                             "BL = (1 - alpha) * Inter-CBL + alpha * Intra-CBL (Xu et al. 2025, eq. 13).")
    parser.add_argument('--balance_t', default=0.9, type=float,
                        help="Threshold t for --loss_fn balance (Xu et al. 2025): splits easy/hard pixels "
                             "for Intra-CBL and is also used in the Inter-CBL convergence check.")
    parser.add_argument('--balance_normalized', nargs='?', const='v1', default='none',
                        choices=['none', 'v1', 'v2'],
                        help="Normalize Inter-CBL and Intra-CBL so it is a actual average, so that "
                             "the weights sum to one (not part of Xu et al. 2025, but could help). "
                             "Without a value it is v1, v2 also fixes images without foreground.")
    parser.add_argument('--balance_fallback_epoch', default=-1, type=int,
                        help="Force InterCBL on if the paper's natural convergence criterion "
                             "hasn't triggered by this epoch. Value -1 means epochs // 2. "
                             "0 disables the fallback (paper-only convergence).")
    parser.add_argument('--balance_inter', choices=['binary', 'multiclass'], default='binary',
                        help="InterCBL (background vs foreground) or InterMCBL (a chain over all classes).")
    parser.add_argument('--opt', choices=["adam", "adamw"], default="adam", help="Optimizer used during training.")
    parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate used during training.")
    parser.add_argument( '--context_slices', default=0, type=int, help="Number of neighboring slices before and after the current slice. 0 keeps 2D behavior.")
    parser.add_argument('--scheduler', choices=["none", "step", "plateau", "cosine"], default="none", help="Learning rate scheduler.")
    parser.add_argument('--scheduler_step_size', default=10, type=int, help="Step size for --scheduler step.")
    parser.add_argument('--scheduler_gamma', default=0.1, type=float,help="LR decay factor for step/plateau schedulers.")
    parser.add_argument('--scheduler_patience', default=5, type=int, help="Patience for --scheduler plateau.")
    parser.add_argument('--scheduler_t_max', default=0, type=int, help="T_max for --scheduler cosine. 0 means use --epochs.")
    parser.add_argument('--early_stopping_patience', default=0, type=int, help="Stop after this many epochs without validation Dice improvement. 0 disables early stopping.")
    parser.add_argument( '--early_stopping_min_delta', default=0.0, type=float,help="Minimum validation Dice improvement needed to reset early stopping.")
    parser.add_argument('--deterministic', action='store_true', help="Enable deterministic PyTorch/CUDA behavior and seeded DataLoader shuffling.")
    parser.add_argument('--seed', default=0, type=int, help="Seed used when --deterministic is set.")
    parser.add_argument('--only_count_flops', action='store_true', help="Estimate FLOPs and exit without training.")
    parser.add_argument('--oversample_foreground', action='store_true', help="Enable foreground oversampling during training.")
    parser.add_argument('--oversample_foreground_percent', default=0.5, type=float, help="Fraction of foreground-containing images to sample when --oversample_foreground is set.")
    parser.add_argument('--augment', action='store_true',
                        help="Turn on augmentation for the training data.")
    parser.add_argument('--drop_empty_slices', type=float, default=0.0,
                        help="Fraction (0-1) of purely-background training slices to randomly "
                             "drop (fixed seed). Default 0 keeps every slice, matching the "
                             "baseline. Validation is never filtered.")

    args = parser.parse_args()
    if (args.hu_min is None) != (args.hu_max is None):
        parser.error('Supply --hu_min and --hu_max together')
    if not (0.0 <= args.drop_empty_slices <= 1.0):
        parser.error('--drop_empty_slices must be between 0 and 1')
    if args.hu_min is not None and not (np.isfinite(args.hu_min) and np.isfinite(args.hu_max)
                                        and args.hu_min < args.hu_max):
        parser.error('HU bounds must be finite, with --hu_min < --hu_max')
    args.hu_windows = parse_hu_windows(parser, args)
    if args.dataset is None:
        args.dataset = 'SEGTHOR' if args.test_pipeline else 'TOY2'
    if args.model is None:
        args.model = datasets_params[args.dataset]['model']
    if args.context_slices < 0:
        parser.error('--context_slices must be 0 or more')
    if args.foundation_model is None:
        if args.foundation_fusion is not None or args.foundation_channels is not None:
            parser.error('--foundation_fusion and --foundation_channels need a --foundation_model')
    else:
        if not args.model.startswith('unet'):
            parser.error(f'--foundation_model only works with the U-Nets, not --model {args.model}')
        if args.foundation_fusion is None:
            parser.error('--foundation_model needs --foundation_fusion encoder or decoder')
        if args.foundation_channels is not None and args.foundation_channels < 1:
            parser.error('--foundation_channels must be at least 1')
    if args.dest is None:
        if not args.test_pipeline:
            parser.error('--dest is required unless --test_pipeline is set')
        args.dest = Path('results') / args.dataset.lower() / 'smoke_run'
    elif args.dest.exists() and not args.test_pipeline:
        parser.error(f'--dest {args.dest} already exists. Pick a new --dest or remove the old one first.')
    if args.test_pipeline:
        args.epochs = 1
        print('Smoke run: one training/validation epoch (--epochs is overridden).')
        if args.dataset == 'SEGTHOR':
            if args.data_dir is None:
                args.data_dir = Path('data/SEGTHOR_smoke')
            ensure_smoke_data(args.data_dir, args.source_dir, args.hu_min, args.hu_max, args.clahe,
                              args.target_spacing, args.fix_aorta_esophagus, args.hu_windows)

    if args.deterministic:
        set_deterministic(args.seed)
    
    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
