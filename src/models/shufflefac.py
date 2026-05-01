from collections import OrderedDict

import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return y * self.sigmoid(x)


class ContextGating(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x * self.sigmoid(y)


class FrequencyAttention(nn.Module):
    def __init__(self, freq_bins):
        super().__init__()
        self.linear = nn.Linear(freq_bins, 1)

    def forward(self, x):
        # x: [B, C, T, F]
        alpha = torch.sigmoid(self.linear(x.mean(dim=2)))
        return alpha.view(x.size(0), x.size(1), 1, 1)


class FrequencyPositionalEncoding(nn.Module):
    def __init__(self, freq_bins):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(freq_bins))

    def forward(self, x):
        # x: [B, C, T, F]
        return self.pos.view(1, 1, 1, -1).expand(x.size(0), x.size(1), x.size(2), -1)


class FACConv(nn.Module):
    def __init__(self, freq_bins):
        super().__init__()
        self.attention = FrequencyAttention(freq_bins)
        self.pos_encoding = FrequencyPositionalEncoding(freq_bins)

    def forward(self, x):
        return x + self.attention(x) * self.pos_encoding(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        if channels % self.groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={self.groups}")
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, channels, height, width)


class ShuffleFACCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        activation="glu",
        conv_dropout=0.2,
        filters=None,
        pooling=None,
        freq_bins=None,
    ):
        super().__init__()
        filters = filters or [16, 32, 64, 128, 128, 128, 128]
        pooling = pooling or [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        freq_bins = freq_bins or [128, 64, 32, 16, 8, 4, 2]
        if not (len(filters) == len(pooling) == len(freq_bins)):
            raise ValueError("filters, pooling, and freq_bins must have the same length")

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
            if conv_dropout and conv_dropout > 0:
                layers[f"dropout{idx}"] = nn.Dropout(conv_dropout)
            layers[f"pooling{idx}"] = nn.AvgPool2d(pooling[idx])

        layers["adaptive_pool"] = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn = nn.Sequential(layers)

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
        return self.cnn(x)


class ShuffleFAC(nn.Module):
    """ShuffleFAC gamma=16 for log-mel input [B, 1, F, T]."""

    def __init__(
        self,
        num_classes,
        in_channels=1,
        n_mels=128,
        activation="glu",
        dropout=0.2,
        filters=None,
    ):
        super().__init__()
        filters = filters or [16, 32, 64, 128, 128, 128, 128]
        pooling = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        freq_bins = []
        current_freq = int(n_mels)
        for pool in pooling:
            freq_bins.append(max(1, current_freq))
            current_freq = max(1, current_freq // int(pool[1]))

        self.filters = filters
        self.cnn = ShuffleFACCNN(
            in_channels=in_channels,
            activation=activation,
            conv_dropout=dropout,
            filters=filters,
            pooling=pooling,
            freq_bins=freq_bins,
        )
        self.fc = nn.Linear(filters[-1], num_classes)

    def forward(self, x):
        # Feature_Extraction_Layer returns [B, 1, F, T]; ShuffleFAC operates on [B, 1, T, F].
        x = x.transpose(2, 3)
        x = self.cnn(x)
        x = x.flatten(1)
        return self.fc(x)
