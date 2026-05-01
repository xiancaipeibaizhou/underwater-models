import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.MultiFeature_Extraction_Layer import MultiFeatureExtractor
from src.models.uatr_knn_graph import UATR_KNN_Graph


class AcousticAuxTargetExtractor(nn.Module):
    """Build normalized handcrafted targets for auxiliary regression."""

    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=20,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        stft_bins=64,
    ):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.stft_bins = stft_bins
        self.aux_target_dim = (n_mfcc * 2) + (n_mfcc * 2) + (stft_bins * 2)
        self.extractor = MultiFeatureExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            disable_mipe=True,
        )

    @torch.no_grad()
    def forward(self, waveform):
        features = self.extractor(waveform)
        mfcc = features["mfcc"]
        delta = features["delta"]
        stft = features["stft"]

        mfcc_target = torch.cat(
            [mfcc.mean(dim=-1), mfcc.std(dim=-1, unbiased=False)],
            dim=-1,
        )
        delta_target = torch.cat(
            [delta.mean(dim=-1), delta.std(dim=-1, unbiased=False)],
            dim=-1,
        )

        stft_mean = stft.mean(dim=-1)
        stft_std = stft.std(dim=-1, unbiased=False)
        stft_mean = F.adaptive_avg_pool1d(
            stft_mean.unsqueeze(1),
            self.stft_bins,
        ).squeeze(1)
        stft_std = F.adaptive_avg_pool1d(
            stft_std.unsqueeze(1),
            self.stft_bins,
        ).squeeze(1)
        stft_target = torch.cat([stft_mean, stft_std], dim=-1)

        aux_target = torch.cat([mfcc_target, delta_target, stft_target], dim=-1)
        return F.layer_norm(aux_target, (aux_target.size(-1),))


class UATR_KNN_REG(nn.Module):
    """UATR_KNN-C classifier with an auxiliary handcrafted-feature head."""

    def __init__(
        self,
        num_classes=5,
        in_channels=1,
        dim=96,
        k=8,
        depth=1,
        dropout=0.2,
        aux_target_dim=208,
        aux_hidden_dim=None,
    ):
        super().__init__()
        aux_hidden_dim = aux_hidden_dim or dim * 2
        self.aux_target_dim = aux_target_dim
        self.backbone = UATR_KNN_Graph(
            num_classes=num_classes,
            in_channels=in_channels,
            dim=dim,
            k=k,
            depth=depth,
            variant="C",
            dropout=dropout,
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden_dim, aux_target_dim),
        )

    def forward(self, x, return_aux=False, extract_feature=False):
        embedding = self.backbone(x, extract_feature=True)
        if extract_feature:
            return embedding

        logits = self.backbone.classifier(embedding)
        if return_aux:
            return logits, self.aux_head(embedding)
        return logits
