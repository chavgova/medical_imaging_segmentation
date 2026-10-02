"""

Implementation of (2D) U-Net based on the original Ronneberger et al., 2015,
the nnU-Net Isensee et al. (2021), and Pytorch-UNet.

"""

from collections import OrderedDict
from typing import Literal, Protocol, get_args

import torch
import torch.nn as nn
from torch import Tensor

Padding = Literal["valid", "same"]
PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
Norm = Literal["none", "batch", "instance"]
Activation = Literal["relu", "leaky_relu", "prelu", "gelu", "silu"]
Downsample = Literal["maxpool", "conv", "strided"]
Upsample = Literal["transpose", "bilinear"]
WeightInit = Literal["kaiming", "pytorch"]
FoundationFusion = Literal["encoder", "decoder"]


class FoundationModel(Protocol):
    """A frozen model that turns the input into a (B, embed_dim, H / patch_size,
    W / patch_size) feature map, e.g. FrozenDino in dino.py.

    This class is not really used in code but is used for type checking
    so that we get an error if we pass a model that's different from this."""

    embed_dim: int
    patch_size: int

    def __call__(self, x: Tensor) -> Tensor: ...


class UNet(nn.Module):
    """

    Implementation of U-Net for medical image segmentation.

    To get the exact architecture that Ronneberger et al. (2015) used
    in the original U-Net paper use the following config:

        in_channels = 1 (their microscopy images are grayscale)
        kernels = 64
        factor = 2
        depth = 4
        kernel_size = 3
        padding = "valid"
        norm = "none"
        activation = "relu"
        downsample = "maxpool"
        upsample = "transpose"
        dropout = the paper uses dropout but gives no rate
        weight_init = "kaiming"

    If you want something more like nnU-Net (Isensee et al., 2021)
    would recommend it, then use the following config:

        in_channels = number of input channels of the data
        kernels = 32
        factor = 2
        max_channels = 512
        depth = until the feature map would get smaller than 4x4, so 6 for 256x256
        kernel_size = 3
        padding = "same"
        padding_mode = "zeros"
        norm = "instance"
        activation = "leaky_relu"
        negative_slope = 0.01
        downsample = "strided"
        upsample = "transpose"
        dropout = 0 (not mentioned in the paper, and off in the nnU-Net code)
        weight_init = "kaiming"

    To get the same model as Pytorch-UNet on GitHub with its default
    bilinear=False use the following config:

        in_channels = number of input channels of the data
        kernels = 64
        factor = 2
        depth = 4
        kernel_size = 3
        padding = "same"
        padding_mode = "zeros"
        norm = "batch"
        activation = "relu"
        downsample = "maxpool"
        upsample = "transpose"
        dropout = 0
        weight_init = "pytorch"

    For their bilinear=True use the same config, but with:

        max_channels = 512
        upsample = "bilinear"

    For a config that's probably good for medical image segmentation on SEGTHOR,
    since it's mostly the defaults that nnU-Net uses:

        in_channels = 5 (2.5D with --context_slices 2)
        kernels = 32 (nnU-Net)
        factor = 2 (you barely see something else being used)
        max_channels = 512 (nnU-Net's cap for 2D)
        depth = 6 (nnU-Net's rule for 256x256)
        kernel_size = 3 (most common)
        padding = "same"
        padding_mode = "zeros"
        norm = "instance" (nnU-Net, does not depend on the batch size)
        activation = "leaky_relu" (nnU-Net)
        negative_slope = 0.01
        downsample = "strided" (nnU-Net)
        upsample = "transpose" (most used)
        dropout = 0 (nnU-Net)
        weight_init = "kaiming"

    You can also use a foundation model like FrozenDino (dino.py) to add
    additional features to the ones learned by U-Net. These features are
    concatenated to the U-Net features at some level of the U that has the
    same resolution. So for instance if the input image is 256x256 and
    the foundation model has a patch size of 8, then we must concatenate
    the features at the level of the U that has is 256 / 8 = 32x32. At
    this level the features are first projected to a smaller (or bigger)
    number of channels with a 1x1 conv, then they are concatenated along
    the channel dimension. Since there are two sides of the U, the foundation
    features can be concatenated either on the encoder side or the decoder
    side.

    References:

        Ronneberger, Fischer & Brox (2015). U-Net: Convolutional Networks for
            Biomedical Image Segmentation. papers/unet_ronneberger_2015.pdf
        Isensee et al. (2021). nnU-Net: a self-configuring method for deep
            learning-based biomedical image segmentation.
            papers/s41592-020-01008-z.pdf
        Pytorch-UNet: https://github.com/milesial/Pytorch-UNet

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: int = 64,
        factor: int = 2,
        max_channels: int | None = None,
        depth: int = 4,
        kernel_size: int = 3,
        padding: Padding = "same",  # for the kernel_size convs, the 2x2 and 1x1 convs need none
        padding_mode: PaddingMode | None = None,
        norm: Norm = "batch",
        activation: Activation = "relu",
        negative_slope: (
            float | None
        ) = None,  # for (initial) slope of leaky relu and prelu
        downsample: Downsample = "maxpool",
        upsample: Upsample = "transpose",
        dropout: float = 0.0,
        weight_init: WeightInit = "kaiming",
        foundation_model: FoundationModel | None = None,
        foundation_channels: int | None = None,  # after the 1x1 conv
        foundation_fusion: FoundationFusion | None = None,
    ):
        super().__init__()
        assert padding in get_args(Padding), f"unknown padding {padding!r}"
        assert norm in get_args(Norm), f"unknown norm {norm!r}"
        assert activation in get_args(Activation), f"unknown activation {activation!r}"
        assert downsample in get_args(Downsample), f"unknown downsample {downsample!r}"
        assert upsample in get_args(Upsample), f"unknown upsample {upsample!r}"
        assert weight_init in get_args(
            WeightInit
        ), f"unknown weight_init {weight_init!r}"
        assert kernel_size % 2 == 1, "kernel_size must be an odd number"
        assert (
            padding_mode is None or padding == "same"
        ), "padding_mode is only used with padding='same'"
        assert padding_mode is None or padding_mode in get_args(
            PaddingMode
        ), f"unknown padding_mode {padding_mode!r}"
        assert (
            max_channels is None or max_channels >= kernels
        ), "max_channels must be at least kernels"
        assert negative_slope is None or activation in (
            "leaky_relu",
            "prelu",
        ), "negative_slope is only used with activation='leaky_relu' or 'prelu'"
        assert in_channels >= 1 and out_channels >= 1, "need at least one channel"
        assert kernels >= 1 and factor >= 1, "kernels and factor must be at least 1"
        assert depth >= 1, "depth must be at least 1"
        assert kernel_size >= 1, "kernel_size must be positive"
        assert 0.0 <= dropout <= 1.0, "dropout must be between 0 and 1"
        assert (foundation_model is None) == (
            foundation_fusion is None
        ), "foundation_fusion must be given exactly when there is a foundation_model"
        assert (
            foundation_model is not None or foundation_channels is None
        ), "foundation_channels is only used with a foundation_model"
        assert foundation_fusion is None or foundation_fusion in get_args(
            FoundationFusion
        ), f"unknown foundation_fusion {foundation_fusion!r}"
        assert (
            foundation_channels is None or foundation_channels >= 1
        ), "foundation_channels must be at least 1"
        if foundation_model is not None:
            assert padding == "same", "a foundation_model needs padding='same'"
            assert isinstance(
                foundation_model, nn.Module
            ), "the foundation_model must be an nn.Module, to be moved and saved with the U-Net"
            assert not any(
                p.requires_grad for p in foundation_model.parameters()
            ), "the foundation_model must be frozen"
            assert (
                isinstance(foundation_model.embed_dim, int)
                and foundation_model.embed_dim >= 1
            ), "the foundation_model's embed_dim must be a positive int"
            patch_size = foundation_model.patch_size
            assert (
                isinstance(patch_size, int)
                and patch_size >= 1
                and patch_size & (patch_size - 1) == 0
            ), "the foundation_model's patch_size must be a power of 2"
            assert (
                patch_size.bit_length() - 1 < depth
            ), f"patch_size {patch_size} needs a level of the U below depth {depth}"

        # PyTorch's defaults: 0.01 for nn.LeakyReLU and 0.25 for nn.PReLU
        if negative_slope is None:
            negative_slope = 0.25 if activation == "prelu" else 0.01

        self.depth: int = depth
        self.kernel_size: int = kernel_size
        self.padding: Padding = padding
        self.padding_mode: PaddingMode = padding_mode or "zeros"
        self.norm: Norm = norm
        self.activation: Activation = activation
        self.negative_slope: float = negative_slope
        self.downsample: Downsample = downsample
        self.upsample: Upsample = upsample
        self.weight_init: WeightInit = weight_init
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        # All layers with weights, collected by the methods that build them so
        # init_weights can initialise them.
        self.convs: list[nn.Conv2d | nn.ConvTranspose2d] = []
        self.norms: list[nn.BatchNorm2d | nn.InstanceNorm2d] = []
        self.prelus: list[nn.PReLU] = []

        # Number of feature channels per level of the U, e.g. 64, 128, 256, 512, 1024.
        # With max_channels=512 that would become 64, 128, 256, 512, 512.
        channels = [kernels * factor**level for level in range(depth + 1)]
        if max_channels is not None:
            channels = [min(c, max_channels) for c in channels]

        # The foundation model's features are concatenated at the level of the U
        # with the same resolution as its patch grid
        self.foundation_model = foundation_model
        self.foundation_fusion: FoundationFusion | None = foundation_fusion
        self.foundation_level: int | None = None
        self.foundation_channels: int | None = None
        self.foundation_embed_dim: int | None = None
        self.foundation_patch_size: int | None = None
        extra = 0
        if foundation_model is not None:
            self.foundation_level = foundation_model.patch_size.bit_length() - 1
            assert 2**self.foundation_level == foundation_model.patch_size
            extra = (
                foundation_channels
                if foundation_channels is not None
                else channels[self.foundation_level] // 2
            )
            assert extra >= 1, "the 1x1 conv would have no output channels"
            self.foundation_channels = extra
            self.foundation_embed_dim = foundation_model.embed_dim
            self.foundation_patch_size = foundation_model.patch_size

        skip_channels = channels[:depth]
        if foundation_fusion == "encoder" and self.foundation_level is not None:
            skip_channels[self.foundation_level] += extra

        # With downsample="strided" (nnU-Net) there is no separate downsampling
        # step, instead the first conv of the next block has a stride of 2.
        stride = 2 if downsample == "strided" else 1

        # Left side of the U (encoder): at every level two convs (e.g. 3x3),
        # followed by a downsampling step (e.g. a 2x2 max pool) that halves the
        # resolution.
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for level in range(depth):
            self.encoder_blocks.append(
                self.double_conv(
                    in_channels if level == 0 else skip_channels[level - 1],
                    channels[level],
                    stride=1 if level == 0 else stride,
                )
            )
            self.downsamplers.append(self.downsampling(skip_channels[level]))

        # Bottom of the U: two convs (e.g. 3x3). The paper puts dropout at
        # the end of the 'contracting path' so presumably after the
        # bottleneck
        self.bottleneck = self.double_conv(
            skip_channels[depth - 1], channels[depth], stride=stride
        )
        self.dropout = nn.Dropout2d(dropout)

        # Right side of the U (decoder): per level an upsampling step (e.g. a 2x2
        # up-conv) that doubles the resolution, then the feature map from the left
        # of the U is stacked, then two regular convs
        if upsample == "bilinear":
            decoder_out = [channels[0]] + channels[: depth - 1]
        else:
            decoder_out = channels[:depth]
        below = decoder_out[1:] + [channels[depth]]
        upsampled = below if upsample == "bilinear" else channels[:depth]

        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(depth)):
            self.upsamplers.append(self.upsampling(below[level], upsampled[level]))
            self.decoder_blocks.append(
                self.double_conv(
                    skip_channels[level]
                    + upsampled[level]
                    + (
                        extra
                        if foundation_fusion == "decoder"
                        and level == self.foundation_level
                        else 0
                    ),
                    decoder_out[level],
                    mid_channels=channels[level],
                )
            )

        # They do one more conv at the end with kernel size 1
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.convs.append(self.output)

        if foundation_model is not None:
            conv = nn.Conv2d(
                foundation_model.embed_dim, extra, kernel_size=1, bias=norm == "none"
            )
            self.convs.append(conv)
            self.foundation_projection = nn.Sequential(
                OrderedDict(
                    conv=conv, norm=self.normalization(extra), act=self.nonlinearity()
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() == 4 and x.size(1) == self.in_channels
        ), f"expected (B, {self.in_channels}, H, W), got {tuple(x.shape)}"
        batch, _, height, width = x.shape
        if self.padding == "same":
            multiple = 2**self.depth
            assert (
                height % multiple == 0 and width % multiple == 0
            ), f"input height and width must be divisible by {multiple}"

        # get features from the foundation model
        features = None
        fused = 0
        if self.foundation_model is not None:
            assert isinstance(self.foundation_model, nn.Module)
            assert (
                self.foundation_channels is not None
                and self.foundation_embed_dim is not None
                and self.foundation_patch_size is not None
            ), "foundation_channels, embed_dim and patch_size must be set with a foundation_model"
            assert (
                not self.foundation_model.training
            ), "the foundation_model must stay in eval mode"
            assert (
                self.foundation_model.embed_dim == self.foundation_embed_dim
                and self.foundation_model.patch_size == self.foundation_patch_size
            ), "the foundation_model's embed_dim or patch_size changed after construction"
            grid = (
                height // self.foundation_patch_size,
                width // self.foundation_patch_size,
            )
            raw = self.foundation_model(x)
            assert raw.shape == (
                batch,
                self.foundation_embed_dim,
                *grid,
            ), f"foundation_model gave {tuple(raw.shape)}, expected {(batch, self.foundation_embed_dim, *grid)}"
            features = self.foundation_projection(raw)
            assert features.shape == (batch, self.foundation_channels, *grid)

        # Left of the U
        skips = []
        for level, (encoder_block, downsampler) in enumerate(
            zip(self.encoder_blocks, self.downsamplers)
        ):
            x = encoder_block(x)
            if (
                features is not None
                and self.foundation_fusion == "encoder"
                and level == self.foundation_level
            ):
                assert (
                    features.shape[2:] == x.shape[2:]
                ), f"features {tuple(features.shape)} don't match level {level} {tuple(x.shape)}"
                x = torch.cat([x, features], dim=1)
                fused += 1
            skips.append(x)
            x = downsampler(x)

        # Bottom of the U
        x = self.dropout(self.bottleneck(x))

        # Right of the U
        for level, upsampler, decoder_block in zip(
            reversed(range(self.depth)), self.upsamplers, self.decoder_blocks
        ):
            x = upsampler(x)
            skip = skips.pop()

            # Copy and crop: the unpadded convs of the original lose border pixels,
            # so the skip is cropped to the centre. With padding="same" both
            # already have the same size and this does nothing.
            top = (skip.size(2) - x.size(2)) // 2
            left = (skip.size(3) - x.size(3)) // 2
            skip = skip[:, :, top : top + x.size(2), left : left + x.size(3)]

            if (
                features is not None
                and self.foundation_fusion == "decoder"
                and level == self.foundation_level
            ):
                assert (
                    features.shape[2:] == x.shape[2:] == skip.shape[2:]
                ), f"features {tuple(features.shape)} don't match level {level} {tuple(x.shape)}"
                x = torch.cat([skip, x, features], dim=1)
                fused += 1
            else:
                x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)

        assert fused == (
            features is not None
        ), "the foundation features must be concatenated exactly once"
        x = self.output(x)
        assert self.padding != "same" or x.shape == (
            batch,
            self.out_channels,
            height,
            width,
        ), f"output {tuple(x.shape)} doesn't match the input {(batch, height, width)}"
        return x

    def double_conv(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,  # first conv only, used for downsample="strided"
        mid_channels: (
            int | None
        ) = None,  # out_channels when None, only differs in the decoder with upsample="bilinear"
    ) -> nn.Sequential:
        """(conv -> norm -> activation) x 2"""
        mid_channels = mid_channels or out_channels
        return nn.Sequential(
            OrderedDict(
                conv1=self.conv(in_channels, mid_channels, stride),
                norm1=self.normalization(mid_channels),
                act1=self.nonlinearity(),
                conv2=self.conv(mid_channels, out_channels),
                norm2=self.normalization(out_channels),
                act2=self.nonlinearity(),
            )
        )

    def downsampling(self, channels: int) -> nn.Module:
        match self.downsample:
            # Original U-Net and Pytorch-UNet
            case "maxpool":
                return nn.MaxPool2d(kernel_size=2)
            # None of the three: a learned 2x2 conv with stride 2, the mirror of
            # the transposed 2x2 up-conv
            case "conv":
                conv = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
                self.convs.append(conv)
                return conv
            # nnU-Net
            case "strided":
                return nn.Identity()  # the next block's first conv downsamples

    def upsampling(self, in_channels: int, out_channels: int) -> nn.Module:
        match self.upsample:
            # Original U-Net ("up-convolution"), nnU-Net and Pytorch-UNet with
            # bilinear=False
            case "transpose":
                up_conv = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                )
                self.convs.append(up_conv)
                return up_conv
            # Pytorch-UNet with bilinear=True
            case "bilinear":
                # No weights, so the channels stay the same (in == out).
                return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def conv(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,  # 2 for the first conv of a block with downsample="strided" (nnU-Net)
            padding=0 if self.padding == "valid" else self.kernel_size // 2,
            padding_mode=self.padding_mode,
            bias=self.norm
            == "none",  # a norm layer right after the conv cancels the bias
        )
        self.convs.append(conv)  # for parameter initialization
        return conv

    def normalization(self, channels: int) -> nn.Module:
        norm: nn.BatchNorm2d | nn.InstanceNorm2d
        match self.norm:
            # Original U-Net
            case "none":
                return nn.Identity()
            # Pytorch-UNet
            case "batch":
                norm = nn.BatchNorm2d(channels)
            # nnU-Net
            case "instance":
                norm = nn.InstanceNorm2d(channels, affine=True)
        self.norms.append(norm)  # for parameter initialization
        return norm

    def nonlinearity(self) -> nn.Module:
        match self.activation:
            # Original U-Net and Pytorch-UNet
            case "relu":
                return nn.ReLU(inplace=True)
            # nnU-Net
            case "leaky_relu":
                return nn.LeakyReLU(self.negative_slope, inplace=True)
            # used by ENet
            case "prelu":
                prelu = nn.PReLU(init=self.negative_slope)
                self.prelus.append(prelu)  # for param initialization
                return prelu
            # others
            case "gelu":
                return nn.GELU()
            case "silu":
                return nn.SiLU(inplace=True)

    def init_weights(self) -> None:
        nonlinearity: Literal["relu", "leaky_relu"] = (
            "leaky_relu" if self.activation in ("leaky_relu", "prelu") else "relu"
        )
        for conv in self.convs:
            match self.weight_init:
                # Original U-Net and nnU-Net
                case "kaiming":
                    nn.init.kaiming_normal_(
                        conv.weight, a=self.negative_slope, nonlinearity=nonlinearity
                    )
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)
                # Pytorch-UNet
                case "pytorch":
                    conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

        for prelu in self.prelus:
            nn.init.constant_(prelu.weight, self.negative_slope)
