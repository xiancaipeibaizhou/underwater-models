"""FA_UATR_KNN 融合模型。

该文件保留后续开发路线：用 FASCStem 提供频率自适应前端，再接
UATR_KNN 风格的 Transformer + KNN-MRGraphConv，并通过标量门控残差
融合 Transformer 输出和图分支输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fasc_stem import FASCStem
from src.models.uatr_knn_graph import MRGraphConv


class FA_UATR_KNN(nn.Module):
    """FA/FASC 前端与 UATR_KNN 关系建模的融合模型。

    输入: Log-Mel `[B, 1, F, T]`。
    输出: 分类 logits `[B, num_classes]`，或 extract_feature=True 时输出
    全局特征 `[B, dim]`。
    """
    def __init__(
        self,
        num_classes,
        in_channels=1,
        dim=128,
        k=8,
        depth=1,
        dropout=0.2,
        max_tokens=1024,
        n_mels=128,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.max_tokens = max_tokens
        self.last_gate_mean = None

        self.fasc_stem = FASCStem(
            in_channels=in_channels,
            dim=dim,
            n_mels=n_mels,
            activation="glu",
            dropout=dropout,
            filters=[16, 32, 64, 128, 128, 128, 128],
        )
        self.token_proj = nn.Identity()
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        nhead = 4 if dim % 4 == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm_trans = nn.LayerNorm(dim)
        self.graph_conv = MRGraphConv(dim=dim, dropout=dropout)
        hidden_gate_dim = max(16, dim // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden_gate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_gate_dim, 1),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def _position_embedding(self, num_tokens):
        """返回与当前 token 数匹配的位置编码，必要时线性插值。"""
        if num_tokens <= self.pos_embed.size(1):
            return self.pos_embed[:, :num_tokens, :]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=num_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def _get_knn_graph(self, x):
        """复用 UATR_KNN 的思路，在单样本内部构建 KNN token 图。"""
        # x: [B, N, C]
        num_tokens = x.size(1)
        if num_tokens <= 1:
            return None
        k = min(self.k, num_tokens - 1)
        dist = torch.cdist(x, x)
        eye = torch.eye(num_tokens, device=x.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        return dist.topk(k, largest=False).indices

    def forward(self, x, extract_feature=False):
        """执行 FASC stem、token 化、Transformer、KNN-GNN 与门控融合。"""
        feature_map = self.fasc_stem(x)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)

        tokens = tokens + self._position_embedding(tokens.size(1)).to(tokens.dtype)
        tokens = self.pos_drop(tokens)

        x_trans = self.transformer(tokens)
        x_trans = self.norm_trans(x_trans)

        knn_idx = self._get_knn_graph(x_trans)
        x_graph = x_trans if knn_idx is None else self.graph_conv(x_trans, knn_idx)

        gate_input = x_trans.mean(dim=1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        self.last_gate_mean = gate.mean().detach()
        gate = gate.unsqueeze(1)
        x_fused = (1.0 - gate) * x_trans + gate * x_graph

        x_pool = x_fused.transpose(1, 2)
        x_global = self.global_pool(x_pool).squeeze(-1)

        if extract_feature:
            return x_global
        return self.classifier(x_global)


class FA_UATR_KNN_V2(nn.Module):
    """FA_UATR_KNN V2 诊断版。

    V2 保留 V1，不覆盖 `FA_UATR_KNN`。主要诊断点：
    - FASCStem 保留更多频率 token，默认 target_freq=4。
    - 默认 parallel，让 Transformer 和 Graph 分支并行接收 token。
    - 默认使用 pre-transform tokens 构图，避免默认从 x_trans 构图。
    - 支持 2D separable learnable positional embedding。
    - 支持 scalar / token / element gate，默认 token-wise gate。
    """

    def __init__(
        self,
        num_classes,
        in_channels=1,
        dim=128,
        k=8,
        depth=1,
        dropout=0.2,
        max_tokens=2048,
        n_mels=128,
        fa_target_freq=4,
        fa_arch="parallel",
        pos_type="2d",
        knn_metric="cosine",
        knn_source="pre_trans",
        gate_type="token",
        gate_init_bias=-2.0,
    ):
        super().__init__()
        if fa_arch not in ("serial", "parallel"):
            raise ValueError("fa_arch must be 'serial' or 'parallel'.")
        if pos_type not in ("1d", "2d"):
            raise ValueError("pos_type must be '1d' or '2d'.")
        if knn_metric not in ("l2", "cosine"):
            raise ValueError("knn_metric must be 'l2' or 'cosine'.")
        if knn_source not in ("pre_trans", "post_trans"):
            raise ValueError("knn_source must be 'pre_trans' or 'post_trans'.")
        if gate_type not in ("scalar", "token", "element"):
            raise ValueError("gate_type must be 'scalar', 'token', or 'element'.")

        self.dim = dim
        self.k = k
        self.max_tokens = max_tokens
        self.fa_arch = fa_arch
        self.pos_type = pos_type
        self.knn_metric = knn_metric
        self.knn_source = knn_source
        self.gate_type = gate_type
        self.last_gate_mean = None

        self.fasc_stem = FASCStem(
            in_channels=in_channels,
            dim=dim,
            n_mels=n_mels,
            activation="glu",
            dropout=dropout,
            filters=[16, 32, 64, 128, 128, 128, 128],
            target_freq=fa_target_freq,
        )
        self.token_proj = nn.Identity()

        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.freq_embed = nn.Parameter(torch.randn(1, dim, max(16, fa_target_freq), 1) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, dim, 1, max_tokens) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        nhead = 4 if dim % 4 == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm_trans = nn.LayerNorm(dim)
        self.graph_conv = MRGraphConv(dim=dim, dropout=dropout)

        gate_out_dim = 1 if gate_type in ("scalar", "token") else dim
        hidden_gate_dim = max(16, dim // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_gate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_gate_dim, gate_out_dim),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, gate_init_bias)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def _position_embedding_1d(self, num_tokens):
        if num_tokens <= self.pos_embed.size(1):
            return self.pos_embed[:, :num_tokens, :]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=num_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def _add_2d_position_embedding(self, feature_map):
        freq_size, time_size = feature_map.size(2), feature_map.size(3)
        freq_embed = self.freq_embed
        if freq_embed.size(2) != freq_size:
            freq_embed = F.interpolate(freq_embed, size=(freq_size, 1), mode="bilinear", align_corners=False)
        time_embed = self.time_embed
        if time_embed.size(3) != time_size:
            time_embed = F.interpolate(time_embed, size=(1, time_size), mode="bilinear", align_corners=False)
        return feature_map + freq_embed.to(feature_map.dtype) + time_embed.to(feature_map.dtype)

    def _get_knn_graph(self, x):
        num_tokens = x.size(1)
        if num_tokens <= 1:
            return None
        k = min(self.k, num_tokens - 1)

        if self.knn_metric == "cosine":
            x_norm = F.normalize(x, p=2, dim=-1)
            sim = torch.matmul(x_norm, x_norm.transpose(1, 2))
            eye = torch.eye(num_tokens, device=x.device, dtype=torch.bool).unsqueeze(0)
            sim = sim.masked_fill(eye, float("-inf"))
            return sim.topk(k, largest=True).indices

        dist = torch.cdist(x, x)
        eye = torch.eye(num_tokens, device=x.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        return dist.topk(k, largest=False).indices

    def _graph_branch(self, graph_source):
        knn_idx = self._get_knn_graph(graph_source)
        if knn_idx is None:
            return graph_source
        return self.graph_conv(graph_source, knn_idx)

    def _fuse(self, x_trans, x_graph):
        gate_input = torch.cat([x_trans, x_graph], dim=-1)
        if self.gate_type == "scalar":
            gate = torch.sigmoid(self.gate_mlp(gate_input.mean(dim=1))).unsqueeze(1)
        else:
            gate = torch.sigmoid(self.gate_mlp(gate_input))
        self.last_gate_mean = gate.mean().detach()
        return x_trans + gate * (x_graph - x_trans)

    def forward(self, x, extract_feature=False):
        feature_map = self.fasc_stem(x)
        if self.pos_type == "2d":
            feature_map = self._add_2d_position_embedding(feature_map)
            tokens = feature_map.flatten(2).transpose(1, 2)
        else:
            tokens = feature_map.flatten(2).transpose(1, 2)
            tokens = tokens + self._position_embedding_1d(tokens.size(1)).to(tokens.dtype)
        tokens = self.pos_drop(self.token_proj(tokens))

        x_trans = self.transformer(tokens)
        x_trans = self.norm_trans(x_trans)

        if self.fa_arch == "parallel":
            graph_source = tokens if self.knn_source == "pre_trans" else x_trans
            x_graph = self._graph_branch(graph_source)
        else:
            graph_source = x_trans if self.knn_source == "post_trans" else tokens
            x_graph = self._graph_branch(graph_source)

        x_fused = self._fuse(x_trans, x_graph)
        x_global = self.global_pool(x_fused.transpose(1, 2)).squeeze(-1)

        if extract_feature:
            return x_global
        return self.classifier(x_global)
