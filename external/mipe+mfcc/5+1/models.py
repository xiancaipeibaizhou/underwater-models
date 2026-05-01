import torch
import torch.nn as nn

# -------------------- 基础组件 --------------------
class ConvBlock(nn.Module):
    """Conv1d + BN + ELU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(self.bn(self.conv(x)))

class ResidualConvBlock(nn.Module):
    """残差卷积块：两个ConvBlock + 残差连接 (对应论文图3)"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size),
            ConvBlock(out_channels, out_channels, kernel_size)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

# -------------------- MRC (Multi-scale Residual Convolution) --------------------
class MRC(nn.Module):
    """多尺度残差卷积模块，三个分支，每个分支三个残差块 + 平均池化(4,4)"""
    def __init__(self, in_channels=1, out_channels=16, kernels=[3,5,7]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            branch = nn.Sequential(
                ResidualConvBlock(in_channels, out_channels, k),
                ResidualConvBlock(out_channels, out_channels, k),
                ResidualConvBlock(out_channels, out_channels, k),
                nn.AvgPool1d(kernel_size=4, stride=4)
            )
            self.branches.append(branch)

    def forward(self, x):
        # x: (batch, in_channels, seq_len) 例如 (batch,1,2000)
        out_branches = [branch(x) for branch in self.branches]  # 每个输出 (batch,16,500)
        out = torch.cat(out_branches, dim=1)                    # (batch,48,500)
        return out

# -------------------- CBAM 1D --------------------
class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention1D(channels, reduction)
        self.spatial_att = SpatialAttention1D()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# -------------------- D_Block (Attention Residual Block) --------------------
class D_Block(nn.Module):
    """CBAM + 残差连接，并调整通道数"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbam = CBAM1D(in_channels)
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        # 残差连接可能需降采样
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.cbam(x)
        out = self.conv1x1(out)
        # 如果序列长度不一致，对identity进行截断或池化（这里假设空间尺寸不变）
        # CBAM不改变空间尺寸，conv1x1也不改变，所以尺寸一致
        return out + identity

# -------------------- 完整模型 --------------------
class MRC_CBAM(nn.Module):
    def __init__(self, input_dim=2000, num_classes=5):
        super().__init__()
        self.input_reshape = nn.Unflatten(1, (1, input_dim))  # (batch, 2000) -> (batch, 1, 2000)
        self.mrc = MRC(in_channels=1, out_channels=16, kernels=[3,5,7])
        self.d_block = D_Block(in_channels=48, out_channels=16)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 16)
        self.classifier = nn.Linear(16, num_classes)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_reshape(x)          # (batch, 1, 2000)
        x = self.mrc(x)                    # (batch, 48, 500)
        x = self.d_block(x)                # (batch, 16, 500)
        x = self.global_pool(x).squeeze(-1)  # (batch, 16)
        x = self.dropout(self.elu(self.fc1(x)))  # (batch, 256)
        x = self.dropout(self.elu(self.fc2(x)))  # (batch, 16)
        return self.classifier(x)          # (batch, num_classes)