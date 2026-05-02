"""Recording-level clip graph built on top of ShuffleFAC embeddings.

The model receives a small bag of clips from the same recording:
`[B, S, 1, F, T]`. Each clip is encoded by ShuffleFAC, then clip embeddings
become graph nodes connected by temporal and/or cosine-similarity edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.shufflefac import ShuffleFAC


class ClipGraphConv(nn.Module):
    """Lightweight message passing over clip nodes.

    `knn_idx` contains neighbor indices for each clip node. The message is the
    mean relative neighbor offset, concatenated with the current node feature.
    """

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, knn_idx):
        # x: [B, S, D], knn_idx: [B, S, K]
        if knn_idx is None or knn_idx.numel() == 0 or knn_idx.size(-1) == 0:
            return torch.zeros_like(x)

        batch_size, num_nodes, dim = x.shape
        num_neighbors = knn_idx.size(-1)
        idx_base = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1) * num_nodes
        flat_idx = (knn_idx + idx_base).reshape(-1)

        x_flat = x.reshape(batch_size * num_nodes, dim)
        neighbors = x_flat[flat_idx].view(batch_size, num_nodes, num_neighbors, dim)
        neighbor_delta = (neighbors - x.unsqueeze(2)).mean(dim=2)
        msg_input = torch.cat([x, neighbor_delta], dim=-1)
        return self.norm(self.msg_mlp(msg_input))


class ShuffleFACClipGraph(nn.Module):
    """Recording-level graph aggregation for ShuffleFAC clip embeddings."""

    def __init__(
        self,
        num_classes,
        in_channels=1,
        gamma=16,
        embed_dim=128,
        graph_hidden_dim=128,
        graph_layers=1,
        graph_k=2,
        dropout=0.2,
        edge_mode="temporal_similarity",
        pooling="attention",
        n_mels=128,
    ):
        super().__init__()
        if edge_mode not in {"temporal", "similarity", "temporal_similarity"}:
            raise ValueError(f"Unsupported edge_mode: {edge_mode}")
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        filters = [gamma, gamma * 2, gamma * 4, gamma * 8, gamma * 8, gamma * 8, gamma * 8]
        encoder_dim = filters[-1]
        self.encoder = ShuffleFAC(
            num_classes=num_classes,
            in_channels=in_channels,
            n_mels=n_mels,
            activation="glu",
            dropout=dropout,
            filters=filters,
        )
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_k = int(graph_k)
        self.edge_mode = edge_mode
        self.pooling = pooling

        self.node_proj = (
            nn.Identity()
            if encoder_dim == graph_hidden_dim
            else nn.Linear(encoder_dim, graph_hidden_dim)
        )
        self.graph_layers = nn.ModuleList(
            [ClipGraphConv(graph_hidden_dim, dropout=dropout) for _ in range(int(graph_layers))]
        )
        self.graph_res_scale = nn.Parameter(torch.tensor(0.1))

        if pooling == "attention":
            self.attn_mlp = nn.Sequential(
                nn.Linear(graph_hidden_dim, graph_hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(graph_hidden_dim // 2, 1),
            )
        else:
            self.attn_mlp = None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_dim, num_classes),
        )

        self.last_attn_entropy = None
        self.last_graph_delta_norm = None

    def _temporal_edges(self, batch_size, num_clips, device):
        prev_idx = torch.arange(num_clips, device=device) - 1
        next_idx = torch.arange(num_clips, device=device) + 1
        center_idx = torch.arange(num_clips, device=device)
        prev_idx = torch.where(prev_idx >= 0, prev_idx, center_idx)
        next_idx = torch.where(next_idx < num_clips, next_idx, center_idx)
        idx = torch.stack([prev_idx, next_idx], dim=-1)
        return idx.unsqueeze(0).expand(batch_size, -1, -1)

    def _similarity_edges(self, nodes):
        batch_size, num_clips, _ = nodes.shape
        if num_clips <= 1 or self.graph_k <= 0:
            return nodes.new_empty(batch_size, num_clips, 0, dtype=torch.long)

        top_k = min(self.graph_k, num_clips - 1)
        nodes_norm = F.normalize(nodes, p=2, dim=-1)
        sim = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))
        eye = torch.eye(num_clips, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, -torch.finfo(sim.dtype).max)
        return sim.topk(top_k, dim=-1, largest=True).indices

    def _build_clip_graph(self, nodes):
        batch_size, num_clips, _ = nodes.shape
        if num_clips <= 1:
            return nodes.new_empty(batch_size, num_clips, 0, dtype=torch.long)

        edge_parts = []
        if self.edge_mode in {"temporal", "temporal_similarity"}:
            edge_parts.append(self._temporal_edges(batch_size, num_clips, nodes.device))
        if self.edge_mode in {"similarity", "temporal_similarity"}:
            edge_parts.append(self._similarity_edges(nodes))
        if not edge_parts:
            return nodes.new_empty(batch_size, num_clips, 0, dtype=torch.long)
        return torch.cat(edge_parts, dim=-1)

    def _pool_graph(self, graph_nodes):
        if self.pooling == "mean":
            self.last_attn_entropy = None
            return graph_nodes.mean(dim=1)

        attn_score = self.attn_mlp(graph_nodes)
        attn_weight = torch.softmax(attn_score, dim=1)
        entropy = -(attn_weight * (attn_weight + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()
        return (attn_weight * graph_nodes).sum(dim=1)

    def forward(self, x, extract_feature=False):
        # x: [B, S, 1, F, T]
        if x.ndim != 5:
            raise ValueError(f"ShuffleFACClipGraph expects [B,S,1,F,T], got {tuple(x.shape)}")

        batch_size, num_clips, channels, freq_bins, time_bins = x.shape
        x_flat = x.reshape(batch_size * num_clips, channels, freq_bins, time_bins)
        clip_emb = self.encoder(x_flat, extract_feature=True)
        nodes = clip_emb.view(batch_size, num_clips, -1)
        nodes = self.node_proj(nodes)

        if num_clips <= 1 or len(self.graph_layers) == 0:
            graph_nodes = nodes
        else:
            knn_idx = self._build_clip_graph(nodes)
            graph_update = nodes
            for graph_layer in self.graph_layers:
                graph_update = graph_layer(graph_update, knn_idx)
            graph_nodes = nodes + self.graph_res_scale * graph_update

        self.last_graph_delta_norm = (graph_nodes - nodes).norm(dim=-1).mean().detach()
        recording_emb = self._pool_graph(graph_nodes)
        if extract_feature:
            return recording_emb
        return self.classifier(recording_emb)
