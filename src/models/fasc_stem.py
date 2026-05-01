"""FASC 前端模块。

从 ShuffleFAC gamma=16 中抽取不含全局池化和分类器的前端，供后续
FA_UATR_KNN 这类 token 模型复用。该模块只输出 feature map。
"""

from collections import OrderedDict

import torch.nn as nn

from src.models.shufflefac import ChannelShuffle, ContextGating, FACConv, GLU


class FASCStem(nn.Module):
    """ShuffleFAC gamma=16 front-end without global pooling or classifier.

    Input is the project-standard log-mel tensor [B, 1, F, T]. Internally this
    follows ShuffleFAC's [B, C, T, F] convention, then returns [B, C, Fp, Tp].
    """

    def __init__(
        self,
        in_channels=1,
        dim=128,
        n_mels=128,
        activation="glu",
        dropout=0.2,
        filters=None,
        target_freq=1,
    ):
        super().__init__()
        if target_freq not in (1, 4, 8):
            raise ValueError("target_freq must be one of {1, 4, 8}.")
        filters = filters or [16, 32, 64, 128, 128, 128, 128]
        pooling_by_target_freq = {
            1: [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)],
            4: [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
            8: [(2, 2), (2, 2), (1, 2), (1, 2), (1, 1), (1, 1), (1, 1)],
        }
        pooling = pooling_by_target_freq[target_freq]
        self.target_freq = target_freq

        freq_bins = []
        current_freq = int(n_mels)
        for pool in pooling:
            freq_bins.append(max(1, current_freq))
            current_freq = max(1, current_freq // int(pool[1]))

        layers = OrderedDict()
        for idx, out_channels in enumerate(filters):
            current_in = in_channels if idx == 0 else filters[idx - 1]
            layers[f"fac_conv{idx}"] = FACConv(freq_bins[idx])
            if idx == 0:
                layers[f"conv{idx}"] = nn.Conv2d(
                    current_in, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                mid_channels = out_channels // 2
                layers[f"pre_conv{idx}"] = nn.Conv2d(
                    current_in, mid_channels, kernel_size=1, groups=2
                )
                layers[f"depthwise_conv{idx}"] = nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channels,
                )
                layers[f"channel_shuffle{idx}"] = ChannelShuffle(groups=2)
                layers[f"pointwise_conv{idx}"] = nn.Conv2d(
                    mid_channels, out_channels, kernel_size=1, groups=2
                )

            layers[f"activation{idx}"] = self._make_activation(activation, out_channels)
            layers[f"batchnorm{idx}"] = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
            if dropout and dropout > 0:
                layers[f"dropout{idx}"] = nn.Dropout(dropout)
            layers[f"pooling{idx}"] = nn.AvgPool2d(pooling[idx])

        self.stem = nn.Sequential(layers)
        self.out_channels = filters[-1]
        self.proj = (
            nn.Identity()
            if self.out_channels == dim
            else nn.Conv2d(self.out_channels, dim, kernel_size=1, bias=False)
        )
        self.out_dim = dim

    @staticmethod
    def _make_activation(name, channels):
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "cg":
            return ContextGating(channels)
        if name == "glu":
            return GLU(channels)
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        # [B, 1, F, T] -> [B, 1, T, F] for ShuffleFAC/FASC operations.
        x = x.transpose(2, 3)
        x = self.stem(x)
        x = self.proj(x)
        # Return to [B, C, Fp, Tp] for tokenization by downstream models.
        return x.transpose(2, 3)
