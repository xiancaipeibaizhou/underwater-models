import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fasc_stem import FASCStem
from src.models.uatr_knn_graph import MRGraphConv


class FA_UATR_KNN(nn.Module):
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
        if num_tokens <= self.pos_embed.size(1):
            return self.pos_embed[:, :num_tokens, :]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=num_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def _get_knn_graph(self, x):
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
