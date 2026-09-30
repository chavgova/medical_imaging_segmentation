"""

Implementation of (2D) U-Net based on the original Ronneberger et al., 2015,
the nnU-Net Isensee et al. (2021), and Pytorch-UNet.

"""

from collections import OrderedDict
from typing import Literal, get_args

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
                    in_channels if level == 0 else channels[level - 1],
                    channels[level],
                    stride=1 if level == 0 else stride,
                )
            )
            self.downsamplers.append(self.downsampling(channels[level]))

        # Bottom of the U: two convs (e.g. 3x3). The paper puts dropout at
        # the end of the 'contracting path' so presumably after the
        # bottleneck
        self.bottleneck = self.double_conv(
            channels[depth - 1], channels[depth], stride=stride
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
                    channels[level] + upsampled[level],  # skip + upsampled
                    decoder_out[level],
                    mid_channels=channels[level],
                )
            )

        # They do one more conv at the end with kernel size 1
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.convs.append(self.output)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == "same":
            multiple = 2**self.depth
            assert (
                x.size(2) % multiple == 0 and x.size(3) % multiple == 0
            ), f"input height and width must be divisible by {multiple}"

        # Left of the U
        skips = []
        for encoder_block, downsampler in zip(self.encoder_blocks, self.downsamplers):
            x = encoder_block(x)
            skips.append(x)
            x = downsampler(x)

        # Bottom of the U
        x = self.dropout(self.bottleneck(x))

        # Right of the U
        for upsampler, decoder_block in zip(self.upsamplers, self.decoder_blocks):
            x = upsampler(x)
            skip = skips.pop()

            # Copy and crop: the unpadded convs of the original lose border pixels,
            # so the skip is cropped to the centre. With padding="same" both
            # already have the same size and this does nothing.
            top = (skip.size(2) - x.size(2)) // 2
            left = (skip.size(3) - x.size(3)) // 2
            skip = skip[:, :, top : top + x.size(2), left : left + x.size(3)]

            x = decoder_block(torch.cat([skip, x], dim=1))

        return self.output(x)

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
