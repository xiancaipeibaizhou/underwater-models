import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.MultiFeature_Extraction_Layer import MultiFeatureExtractor
from src.models.uatr_knn_graph import MRGraphConv


class _Feature2DVectorEncoder(nn.Module):
    def __init__(self, in_channels, d_model, dropout=0.2):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Feature2DTokenEncoder(nn.Module):
    def __init__(self, in_channels, d_model, token_grid, dropout=0.2):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(token_grid),
            nn.Dropout2d(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class _TemporalTokenEncoder(nn.Module):
    def __init__(self, in_dim, d_model, num_tokens=64, dropout=0.2):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: [B, C, T] -> [B, N, C] -> [B, N, d]
        x = F.adaptive_avg_pool1d(x, self.num_tokens).transpose(1, 2)
        return self.proj(x)


class _MultiFeatureModelBase(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=20,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        mipe_m=3,
        mipe_tau=1,
        mipe_c=10,
        mipe_scale=10,
        disable_mipe=False,
        require_cached_mipe=False,
    ):
        super().__init__()
        self.use_mipe = not disable_mipe
        self.mipe_scale = mipe_scale
        self.extractor = MultiFeatureExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            mipe_m=mipe_m,
            mipe_tau=mipe_tau,
            mipe_c=mipe_c,
            mipe_scale=mipe_scale,
            disable_mipe=disable_mipe,
            require_cached_mipe=require_cached_mipe,
        )


class MultiFeatureConcatMLP(_MultiFeatureModelBase):
    """MF_CONCAT: pooled handcrafted features followed by an MLP classifier."""

    def __init__(
        self,
        num_classes,
        sample_rate=16000,
        n_mfcc=20,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        mipe_m=3,
        mipe_tau=1,
        mipe_c=10,
        mipe_scale=10,
        disable_mipe=False,
        require_cached_mipe=False,
        stft_pool_bins=64,
        dropout=0.2,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            mipe_m=mipe_m,
            mipe_tau=mipe_tau,
            mipe_c=mipe_c,
            mipe_scale=mipe_scale,
            disable_mipe=disable_mipe,
            require_cached_mipe=require_cached_mipe,
        )
        self.stft_pool_bins = stft_pool_bins
        in_dim = (n_mfcc * 2) + (n_mfcc * 2) + (stft_pool_bins * 2)
        if self.use_mipe:
            in_dim += mipe_scale

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveform):
        features = self.extractor(waveform)

        mfcc = features["mfcc"]
        delta = features["delta"]
        stft = features["stft"]

        mfcc_feat = torch.cat([mfcc.mean(dim=-1), mfcc.std(dim=-1, unbiased=False)], dim=-1)
        delta_feat = torch.cat([delta.mean(dim=-1), delta.std(dim=-1, unbiased=False)], dim=-1)

        stft_mean = stft.mean(dim=-1)
        stft_std = stft.std(dim=-1, unbiased=False)
        stft_mean = F.adaptive_avg_pool1d(stft_mean.unsqueeze(1), self.stft_pool_bins).squeeze(1)
        stft_std = F.adaptive_avg_pool1d(stft_std.unsqueeze(1), self.stft_pool_bins).squeeze(1)
        stft_feat = torch.cat([stft_mean, stft_std], dim=-1)

        fused = [mfcc_feat, delta_feat, stft_feat]
        if self.use_mipe:
            fused.append(features["mipe"])
        return self.classifier(torch.cat(fused, dim=-1))


class MultiFeatureBranchFusion(_MultiFeatureModelBase):
    """MF_BRANCH: STFT, MFCC/Delta, and optional MIPE branches."""

    def __init__(
        self,
        num_classes,
        sample_rate=16000,
        n_mfcc=20,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        mipe_m=3,
        mipe_tau=1,
        mipe_c=10,
        mipe_scale=10,
        disable_mipe=False,
        require_cached_mipe=False,
        d_model=128,
        dropout=0.2,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            mipe_m=mipe_m,
            mipe_tau=mipe_tau,
            mipe_c=mipe_c,
            mipe_scale=mipe_scale,
            disable_mipe=disable_mipe,
            require_cached_mipe=require_cached_mipe,
        )
        self.stft_branch = _Feature2DVectorEncoder(1, d_model, dropout=dropout)
        self.mfcc_branch = _Feature2DVectorEncoder(2, d_model, dropout=dropout)
        if self.use_mipe:
            self.mipe_branch = nn.Sequential(
                nn.LayerNorm(mipe_scale),
                nn.Linear(mipe_scale, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        branch_count = 3 if self.use_mipe else 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * branch_count),
            nn.Linear(d_model * branch_count, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, waveform):
        features = self.extractor(waveform)
        stft_vec = self.stft_branch(features["stft"].unsqueeze(1))
        mfcc_delta = torch.stack([features["mfcc"], features["delta"]], dim=1)
        mfcc_vec = self.mfcc_branch(mfcc_delta)

        branches = [stft_vec, mfcc_vec]
        if self.use_mipe:
            branches.append(self.mipe_branch(features["mipe"]))
        return self.classifier(torch.cat(branches, dim=-1))


class MultiFeatureCrossAttention(_MultiFeatureModelBase):
    """MF_CROSSATTN: a one-layer CLS query over multi-feature tokens."""

    def __init__(
        self,
        num_classes,
        sample_rate=16000,
        n_mfcc=20,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        mipe_m=3,
        mipe_tau=1,
        mipe_c=10,
        mipe_scale=10,
        disable_mipe=False,
        require_cached_mipe=False,
        d_model=128,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            mipe_m=mipe_m,
            mipe_tau=mipe_tau,
            mipe_c=mipe_c,
            mipe_scale=mipe_scale,
            disable_mipe=disable_mipe,
            require_cached_mipe=require_cached_mipe,
        )
        self.stft_tokens = _Feature2DTokenEncoder(1, d_model, token_grid=(8, 16), dropout=dropout)
        self.mfcc_tokens = _Feature2DTokenEncoder(2, d_model, token_grid=(4, 16), dropout=dropout)
        if self.use_mipe:
            self.mipe_norm = nn.LayerNorm(mipe_scale)
            self.mipe_proj = nn.Linear(1, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.token_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, waveform):
        features = self.extractor(waveform)
        stft_tokens = self.stft_tokens(features["stft"].unsqueeze(1))
        mfcc_delta = torch.stack([features["mfcc"], features["delta"]], dim=1)
        mfcc_tokens = self.mfcc_tokens(mfcc_delta)

        tokens = [stft_tokens, mfcc_tokens]
        if self.use_mipe:
            mipe = self.mipe_norm(features["mipe"]).unsqueeze(-1)
            tokens.append(self.mipe_proj(mipe))
        all_tokens = self.token_norm(torch.cat(tokens, dim=1))

        cls = self.cls_token.expand(features["mfcc"].size(0), -1, -1)
        attn_out, _ = self.cross_attn(
            query=self.query_norm(cls),
            key=all_tokens,
            value=all_tokens,
            need_weights=False,
        )
        cls = self.out_norm(cls + self.drop(attn_out)).squeeze(1)
        return self.classifier(cls)


class MultiViewKNNFusion(_MultiFeatureModelBase):
    """MF_KNN: per-sample multi-view token graphs with MRGraphConv."""

    def __init__(
        self,
        num_classes,
        sample_rate=16000,
        n_mfcc=20,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        mipe_m=3,
        mipe_tau=1,
        mipe_c=10,
        mipe_scale=10,
        disable_mipe=False,
        require_cached_mipe=False,
        d_model=128,
        k=4,
        dropout=0.2,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            mipe_m=mipe_m,
            mipe_tau=mipe_tau,
            mipe_c=mipe_c,
            mipe_scale=mipe_scale,
            disable_mipe=disable_mipe,
            require_cached_mipe=require_cached_mipe,
        )
        self.k = max(1, k)
        self.mfcc_tokens = _TemporalTokenEncoder(n_mfcc, d_model, num_tokens=64, dropout=dropout)
        self.delta_tokens = _TemporalTokenEncoder(n_mfcc, d_model, num_tokens=64, dropout=dropout)
        self.stft_tokens = _Feature2DTokenEncoder(1, d_model, token_grid=(8, 16), dropout=dropout)

        self.mfcc_graph = MRGraphConv(dim=d_model, dropout=dropout)
        self.delta_graph = MRGraphConv(dim=d_model, dropout=dropout)
        self.stft_graph = MRGraphConv(dim=d_model, dropout=dropout)

        if self.use_mipe:
            self.mipe_branch = nn.Sequential(
                nn.LayerNorm(mipe_scale),
                nn.Linear(mipe_scale, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )

        view_count = 4 if self.use_mipe else 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * view_count),
            nn.Linear(d_model * view_count, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _get_knn_graph(self, x):
        batch_size, num_tokens, _ = x.shape
        if num_tokens <= 1:
            return torch.zeros(batch_size, num_tokens, 1, device=x.device, dtype=torch.long)

        k = min(self.k, num_tokens - 1)
        dist = torch.cdist(x, x)
        eye = torch.eye(num_tokens, device=x.device).unsqueeze(0)
        dist = dist + eye * 1e6
        return dist.topk(k, largest=False).indices

    def _graph_pool(self, tokens, graph_layer):
        if tokens.size(1) > 1:
            tokens = graph_layer(tokens, self._get_knn_graph(tokens))
        return tokens.mean(dim=1)

    def forward(self, waveform):
        features = self.extractor(waveform)
        mfcc_tokens = self.mfcc_tokens(features["mfcc"])
        delta_tokens = self.delta_tokens(features["delta"])
        stft_tokens = self.stft_tokens(features["stft"].unsqueeze(1))

        z_mfcc = self._graph_pool(mfcc_tokens, self.mfcc_graph)
        z_delta = self._graph_pool(delta_tokens, self.delta_graph)
        z_stft = self._graph_pool(stft_tokens, self.stft_graph)

        views = [z_mfcc, z_delta, z_stft]
        if self.use_mipe:
            views.append(self.mipe_branch(features["mipe"]))
        return self.classifier(torch.cat(views, dim=-1))
