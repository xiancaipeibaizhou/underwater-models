import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=160,
        n_mels=64,
        fmin=50,
        fmax=8000,
    ):
        super(MelSpectrogramExtractor, self).__init__()

        fmax = min(float(fmax), float(sample_rate) / 2.0)
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=float(fmin),
            f_max=fmax,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            center=True,
            pad_mode="reflect",
            normalized=False,
            norm=None,
            mel_scale="htk",
        )
        self.bn0 = nn.BatchNorm2d(n_mels)

    def forward(self, waveform):
        mel = self.mel_extractor(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-10))

        # BatchNorm over mel bins, preserving the historical output [B, mel, time].
        x = log_mel.unsqueeze(1).transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2).squeeze(1)
        return x
