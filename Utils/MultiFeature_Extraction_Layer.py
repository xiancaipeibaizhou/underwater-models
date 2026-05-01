import math

import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as T


class MultiFeatureExtractor(nn.Module):
    """Extract MFCC, Delta MFCC, log-STFT, and MIPE from raw waveforms."""

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
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mipe_m = mipe_m
        self.mipe_tau = mipe_tau
        self.mipe_c = mipe_c
        self.mipe_scale = mipe_scale
        self.disable_mipe = disable_mipe
        self.require_cached_mipe = require_cached_mipe

        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "center": True,
                "power": 2.0,
            },
        )
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
            center=True,
        )

    def forward(self, x, cached_mipe=None):
        if isinstance(x, dict):
            cached_mipe = x.get("mipe", x.get("cached_mipe", cached_mipe))
            x = x.get("waveform", x.get("x"))
            if x is None:
                raise ValueError("MultiFeatureExtractor input dict must contain 'waveform'.")

        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"MultiFeatureExtractor expects [B, T], got {tuple(x.shape)}")

        mfcc = self.mfcc(x)
        delta_mfcc = AF.compute_deltas(mfcc)
        log_stft = torch.log1p(self.spectrogram(x))
        mipe = self._get_mipe(x, cached_mipe)

        return {
            "mfcc": mfcc,
            "delta": delta_mfcc,
            "stft": log_stft,
            "mipe": mipe,
        }

    def _get_mipe(self, x, cached_mipe=None):
        if self.disable_mipe:
            return x.new_zeros(x.size(0), self.mipe_scale)

        if cached_mipe is not None:
            cached_mipe = cached_mipe.to(device=x.device, dtype=x.dtype)
            if cached_mipe.dim() == 1:
                cached_mipe = cached_mipe.unsqueeze(0)
            if cached_mipe.shape != (x.size(0), self.mipe_scale):
                raise ValueError(
                    f"cached_mipe must have shape {(x.size(0), self.mipe_scale)}, "
                    f"got {tuple(cached_mipe.shape)}"
                )
            return cached_mipe

        if self.require_cached_mipe:
            raise RuntimeError(
                "use_cached_mipe=True requires Dataset to provide cached MIPE. "
                "Run with --precompute_mipe or let the Dataset create .mipe.pt files first."
            )

        return self._batch_mipe(x)

    @torch.no_grad()
    def _batch_mipe(self, x):
        batch_size = x.size(0)
        if self.disable_mipe:
            return x.new_zeros(batch_size, self.mipe_scale)

        mipe_values = []
        for batch_idx in range(batch_size):
            wav = x[batch_idx].detach()
            scale_values = [
                self._ipe_for_scale(wav, scale)
                for scale in range(1, self.mipe_scale + 1)
            ]
            mipe_values.append(torch.stack(scale_values))
        return torch.stack(mipe_values, dim=0).to(dtype=x.dtype)

    def _ipe_for_scale(self, waveform, scale):
        coarse = self._coarse_grain(waveform, scale)
        length_d = coarse.numel()
        num_patterns = length_d - self.mipe_tau * (self.mipe_m - 1)
        if num_patterns <= 0:
            return waveform.new_tensor(0.0)

        mean = coarse.mean()
        std = coarse.std(unbiased=False).clamp_min(1e-6)
        data = 0.5 * (1.0 + torch.erf((coarse - mean) / (std * math.sqrt(2.0))))

        min_data = data.min()
        max_data = data.max()
        data = (data - min_data) / (max_data - min_data).clamp_min(1e-6)

        min_data = data.min()
        max_data = data.max()
        delta_dist = ((max_data - min_data) / self.mipe_c).clamp_min(1e-6)

        first = torch.floor((data[:num_patterns] - min_data) / delta_dist).to(torch.long) + 1
        first = first.clamp(1, self.mipe_c)

        pattern_cols = [first]
        anchor = data[:num_patterns]
        for delay_idx in range(1, self.mipe_m):
            delayed = data[
                delay_idx * self.mipe_tau : delay_idx * self.mipe_tau + num_patterns
            ]
            delta_symbol = torch.floor((delayed - anchor) / delta_dist).to(torch.long)
            pattern_cols.append(first + delta_symbol)

        patterns = torch.stack(pattern_cols, dim=1)
        valid = ((patterns >= 1) & (patterns <= self.mipe_c)).all(dim=1)
        valid_patterns = patterns[valid]

        pattern_count = self.mipe_c ** self.mipe_m
        if pattern_count <= 1:
            return waveform.new_tensor(0.0)

        if valid_patterns.numel() == 0:
            return waveform.new_tensor(0.0)

        bases = torch.tensor(
            [self.mipe_c ** i for i in range(self.mipe_m)],
            device=waveform.device,
            dtype=torch.long,
        )
        pattern_idx = ((valid_patterns - 1) * bases).sum(dim=1)
        counts = torch.bincount(pattern_idx, minlength=pattern_count).to(dtype=waveform.dtype)
        probs = counts / float(num_patterns)
        probs = probs[probs > 0]
        if probs.numel() == 0:
            return waveform.new_tensor(0.0)
        entropy = -(probs * torch.log(probs)).sum()
        return entropy / math.log(pattern_count)

    def _coarse_grain(self, waveform, scale):
        if scale <= 1:
            return waveform
        usable_len = (waveform.numel() // scale) * scale
        if usable_len == 0:
            return waveform.new_empty(0)
        return waveform[:usable_len].reshape(-1, scale).mean(dim=1)

class MultiFeature_Extraction_Layer(MultiFeatureExtractor):
    """Backward-compatible class name matching the file name."""

    pass
