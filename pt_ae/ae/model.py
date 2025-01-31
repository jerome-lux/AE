import torch
import torch.nn as nn
import typing


from .ops import *
from .utils import *
from .norm import LayerNorm2d


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2

    Args:
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        head_init_scale=1.0,
        build_head=True,
        activation="silu",
    ):
        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0], eps=1e-6))
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, _ in enumerate(depths):
            stage = nn.Sequential(
                *[
                    ConvNextBlock(in_channels=dims[i], activation=activation, drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward_features(self, x):
        for i, _ in enumerate(self.depths):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.build_head:
            x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


class Resnet(nn.Module):
    """
    Resnet with classic conv redidual block
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_filters=96,
        stem_kernel_size=7,
        stem_stride=2,
        stem_act="silu",
        stem_norm="bn",
        stem_use_bias=False,
        activation="silu",
        norm="bn",
        fc_layers=[],
        groups=1,
        use_bias=False,
        build_head=False,
        norm_kwargs={},
        dropout=0.5,
    ):

        if not isinstance(use_bias, (tuple, list)):
            use_bias = [use_bias] * 2
        if not isinstance(norm, (tuple, list)):
            norm = [norm] * 2
        if not isinstance(activation, (tuple, list)):
            activation = [activation] * 2

        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.stem = ConvLayer(
            in_channels,
            stem_filters,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            norm=stem_norm,
            activation=stem_act,
            norm_kwargs=norm_kwargs,
            use_bias=stem_use_bias,
        )

        self.ops = nn.ModuleList()
        for i, depth in enumerate(depths):
            for j in range(depth):
                stride = 2 if j == 0 else 1
                if i == 0 and j == 0:
                    input_channels = stem_filters
                elif j == 0 and i > 0:
                    input_channels = dims[i - 1]
                else:
                    input_channels = dims[i]
                self.ops.append(
                    ResBlock(
                        input_channels,
                        dims[i],
                        nconv=2,
                        stride=stride,
                        norm=norm,
                        use_bias=use_bias,
                        activation=activation,
                        norm_kwargs=norm_kwargs,
                        groups=groups,
                    )
                )
        self.head = nn.ModuleList()
        if fc_layers is not None:
            for depth in fc_layers:
                self.head.append(nn.Linear(dims[-1], depth, bias=True))
                self.head.append(build_activation(activation[1]))
                self.head.append(nn.Dropout(dropout))
                dims[-1] = depth

        self.head.append(nn.Linear(dims[-1], num_classes))

    def forward_features(self, x):
        x = self.stem(x)
        for op in self.ops:
            x = op(x)
        return x

    def forward_head(self, x):
        for op in self.head:
            x = op(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.build_head:
            x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.forward_head(x)
        return x


class Resnetb(nn.Module):
    """
    Resnet with bottleneck redidual block
    can be used to make a resnext when groups is > 1
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        groups=1,
        bottleneck_ratio=4,
        se_block=False,
        se_ratio=4,
        stem_filters=96,
        stem_kernel_size=7,
        stem_stride=2,
        activation="silu",
        norm="bn",
        build_head=False,
        norm_kwargs={},
    ):
        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_filters, kernel_size=stem_kernel_size, stride=stem_stride),
            build_norm(norm, stem_filters, **norm_kwargs),
            build_activation(activation),
        )
        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            for j in range(depth):
                stride = 2 if j == 0 else 1
                if i == 0 and j == 0:
                    input_channels = stem_filters
                elif j == 0 and i > 0:
                    input_channels = dims[i - 1]
                else:
                    input_channels = dims[i]
                self.stages.append(
                    BottleneckBlock(
                        input_channels,
                        dims[i],
                        stride=stride,
                        norm=norm,
                        activation=activation,
                        groups=groups,
                        se=se_block,
                        se_ratio=se_ratio,
                        bottleneck_ratio=bottleneck_ratio,
                        norm_kwargs=norm_kwargs,
                    )
                )

        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        for op in self.stages:
            x = op(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.build_head:
            x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.head(x)
        return x


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels,
        latent_dim,
        block: nn.Module,
        stem_kernel_size=3,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_activation="silu",
        stem_norm="bn",
        block_args: dict = {},
        downsampling: str = "conv",
        down_shortcut=True,
    ):
        super().__init__()

        self.op_list = nn.ModuleList()
        self.stem_conv = ConvLayer(
            in_channels, dims[0], kernel_size=stem_kernel_size, norm=stem_norm, activation=stem_activation
        )

        for stage, repeats in enumerate(depths):
            input_channels = dims[stage] if stage == 0 else dims[stage - 1]

            if downsampling.lower() == "conv":
                # depthwise conv stride 2 for downsampling
                DownsamplingBlock = ConvLayer(
                    input_channels,
                    dims[stage],
                    groups=input_channels,
                    kernel_size=3,
                    stride=2,
                    use_bias=True,
                    norm=None,
                    activation=None,
                )

            elif downsampling.lower() == "convpixelunshuffledownsample":
                # or pixel unshuffling
                DownsamplingBlock = ConvPixelUnshuffleDownSampleLayer(
                    input_channels, dims[stage], kernel_size=3, factor=2
                )

            if down_shortcut:
                DownsamplingBlock = BuildResidual(
                    main=DownsamplingBlock,
                    shortcut=PixelUnshuffleChannelAveragingDownSampleLayer(input_channels, dims[stage], factor=2),
                )

            self.op_list.append(DownsamplingBlock)

            for i in range(repeats):
                self.op_list.append(block(in_channels=dims[stage], out_channels=dims[stage], stride=1, **block_args))

        # Project to latent space
        self.bottleneck = nn.Conv2d(dims[-1], latent_dim, 1, bias=True)

    def forward(self, x):

        x = self.stem_conv(x)
        for op in self.op_list:
            x = op(x)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):

    def __init__(
        self,
        out_channels,
        latent_dim,
        block: nn.Module,
        depths=[3, 6, 3, 3],
        dims=[1024, 512, 256, 1024],
        block_args: dict = {},
        upsampling: str = "InterpolateConvUpSample",
        up_shortcut=True,
    ):
        super().__init__()

        self.op_list = nn.ModuleList()
        for stage, repeats in enumerate(depths):
            input_channels = latent_dim if stage == 0 else dims[stage - 1]

            if upsampling.lower() == "interpolateconvupsample":
                UpsamplingBlock = InterpolateConvUpSampleLayer(input_channels, dims[stage], kernel_size=3, factor=2)

            elif upsampling.lower() == "convpixelshuffleupsamplelayer":
                UpsamplingBlock = ConvPixelShuffleUpSampleLayer(input_channels, dims[stage], kernel_size=3, factor=2)

            if up_shortcut:
                shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer(input_channels, dims[stage], factor=2)
                UpsamplingBlock = BuildResidual(main=UpsamplingBlock, shortcut=shortcut)

            self.op_list.append(UpsamplingBlock)

            for i in range(repeats):

                self.op_list.append(block(in_channels=dims[stage], out_channels=dims[stage], stride=1, **block_args))

        # Project back to in_channels
        self.projet = ConvLayer(dims[-1], out_channels, 1, use_bias=True, activation=None, norm=None)

    def forward(self, x):
        for op in self.op_list:
            x = op(x)
        x = self.projet(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
