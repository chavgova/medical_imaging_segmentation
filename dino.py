"""

Frozen DINOv3 or MedDINOv3 which turns a CT-slice into a
(B, dino_channels, H / patch_size, W / patch_size) feature map.
Used as an extension to U-Net, where we concatenate the
DINO feature maps to the features found by U-Net at some
level in the U-Net model where the feature map is also
around (B, unet_channels, H / patch_size, W / patch_size),
ending up with a feature map of shape
(B, unet_channels + dino_channels, H / patch_size, W / patch_size).

References:

    Siméoni et al. (2025). DINOv3. papers/dinov3_simeoni_2025.pdf
        https://arxiv.org/abs/2508.10104
        https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
    Li et al. (2025). MedDINOv3: How to adapt vision foundation models for
        medical image segmentation? papers/meddinov3_li_2025.pdf
        https://arxiv.org/abs/2509.02379
        https://github.com/ricklisz/MedDINOv3

"""

from typing import Any, Literal, get_args

import torch
from huggingface_hub import hf_hub_download
from torch import nn, Tensor
from transformers import AutoModel

# The DINO variants we might be interested in using. The first two are
# the official DINOv3 models, trained on natural images (LVD-1689M). ViT-S/16
# with 21M parameters and 384 channels per token, and ViT-B/16 with 86M and 768.
# Both have 12 layers, patches of 16x16 pixels and 4 register tokens. The last
# is MedDINOv3: the ViT-B/16 trained further on 3.87M CT slices (CT-3M).
Dino = Literal["dinov3-vits16", "dinov3-vitb16", "meddinov3-vitb16"]

# The HU window of our exp_P1_HU data (slice_segthor.py --hu_min -1000 --hu_max 300),
# which stores it as 0 to 1. Only for MedDINOv3, which normalizes the raw HU values,
# so these turn our inputs back into HU. DINOv3 takes the 0 to 1 values as they are.
HU_MIN, HU_MAX = -1000, 300


class FrozenDino(nn.Module):
    def __init__(self, dino: Dino):
        super().__init__()
        assert dino in get_args(Dino), f"unknown dino {dino!r}"
        self.dino: Dino = dino
        match dino:
            # Note that you need Hugging Face authentication and authorization to use these.
            case "dinov3-vits16" | "dinov3-vitb16":
                self.model: Any = AutoModel.from_pretrained(
                    f"facebook/{dino}-pretrain-lvd1689m"
                )
                self.embed_dim: int = self.model.config.hidden_size
                self.patch_size: int = self.model.config.patch_size
                self.register_tokens: int = self.model.config.num_register_tokens
            # Needs the DINOv3 code bundled with MedDINOv3 (the package dinov3)
            case "meddinov3-vitb16":
                # Has to be pip installed from their Github:
                # pip install --no-deps "git+https://github.com/ricklisz/MedDINOv3.git@a2460c2547899283446e520b84523454f7e25608#subdirectory=nnUNet/nnunetv2/training/nnUNetTrainer/dinov3"
                # I import it here so everyone else can still run main.py without having to install dinov3.
                from dinov3.models.vision_transformer import vit_base  # type: ignore

                self.model = vit_base(
                    drop_path_rate=0.0,
                    layerscale_init=1.0e-05,
                    n_storage_tokens=4,
                    qkv_bias=False,
                    mask_k_bias=True,
                )
                path = hf_hub_download(
                    "ricklisz123/MedDINOv3-ViTB-16-CT-3M", "model.pth"
                )
                teacher = torch.load(path, map_location="cpu", weights_only=True)[
                    "teacher"
                ]
                assert all(
                    k.startswith("backbone.") for k in teacher
                ), "unexpected keys in the MedDINOv3 checkpoint"
                self.model.load_state_dict(
                    {k.removeprefix("backbone."): v for k, v in teacher.items()},
                    strict=True,
                )
                self.embed_dim = self.model.embed_dim
                self.patch_size = self.model.patch_size
                self.register_tokens = self.model.n_storage_tokens
        expected = 384 if dino == "dinov3-vits16" else 768
        assert (
            self.embed_dim == expected
        ), f"{dino} should have {expected} channels, not {self.embed_dim}"
        assert self.patch_size == 16, f"{dino} should have 16x16 patches"
        assert self.register_tokens == 4, f"{dino} should have 4 register tokens"
        self.model.requires_grad_(False)
        self.eval()
        assert not any(p.requires_grad for p in self.parameters())
        assert not self.training and not self.model.training

    def train(self, mode: bool = True) -> "FrozenDino":
        # DINOv3 is frozen so eval mode even during training.
        # I think they might do dropout and something with
        # the position embeddings during training for
        # regularization, so we don't want to do that.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        assert (
            not self.training and not self.model.training
        ), "FrozenDino must stay in eval mode"
        assert (
            x.dim() == 4 and x.size(1) % 2 == 1
        ), f"expected (B, 2 * context_slices + 1, H, W), got {tuple(x.shape)}"
        batch, _, height, width = x.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"height and width must be divisible by {self.patch_size}"
        assert (
            x.min() >= 0 and x.max() <= 1
        ), "expected images scaled to [0, 1] (our PNGs divided by 255)"
        grid = (height // self.patch_size, width // self.patch_size)
        center = x.size(1) // 2  # the middle slice with --context_slices
        image = x[:, center : center + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            match self.dino:
                case "dinov3-vits16" | "dinov3-vitb16":
                    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    tokens = self.model(
                        pixel_values=(image - mean) / std
                    ).last_hidden_state
                    assert tokens.shape == (
                        batch,
                        1 + self.register_tokens + grid[0] * grid[1],
                        self.embed_dim,
                    ), f"unexpected tokens {tuple(tokens.shape)}"
                    patches = tokens[:, 1 + self.register_tokens :]
                # The normalize() of MedDINOv3's inference/demo.ipynb: HU clamped
                # to [-1000, 1000], then (HU - mean) / std, repeated to 3 channels
                case "meddinov3-vitb16":
                    hu = (HU_MIN + (HU_MAX - HU_MIN) * image).clamp(-1000, 1000)
                    rgb = ((hu - 65.1084213256836) / 178.01663208007812).expand(
                        -1, 3, -1, -1
                    )
                    patches = self.model.forward_features(rgb)["x_norm_patchtokens"]
        assert patches.shape == (
            batch,
            grid[0] * grid[1],
            self.embed_dim,
        ), f"unexpected patch tokens {tuple(patches.shape)}"
        features = patches.float().transpose(1, 2).unflatten(2, grid)
        assert features.shape == (batch, self.embed_dim, *grid)
        return features
