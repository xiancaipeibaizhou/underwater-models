import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

class dataset(Dataset):
    def __init__(self, file_path, transform=None, class_name_to_id: Optional[Dict[str, int]] = None,
                 mel_kwargs: Optional[Dict[str, Any]] = None):
        self.data_dir = Path(file_path)
        self.transform = transform
        self.meta = []
        self.class_name_to_id = class_name_to_id or {"Cargo": 0, "Passengership": 1, "Tanker": 2, "Tug": 3}
        self.mel_kwargs = mel_kwargs

        if self.data_dir.exists() and self.data_dir.is_dir():
            self.scan_files_with_labels()

    def load_audio(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate



    def waveform_to_log_mel(self,
                             waveform: torch.Tensor,
                             sample_rate: int,
                             n_mels: int = 128,
                             n_fft: int = 4096,
                             hop_length: int = 2048,
                             win_length: int = 4096,
                             f_min: float = 0.0,
                             f_max: float = 8000,
                             power: float = 1,
                             ):
      
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            center=True,
            norm=None,
            mel_scale="htk",
            window_fn=torch.hamming_window
        )
        mels = mel_transform(waveform)
        stype = "magnitude" if power == 1 else "power"
        amp_to_db = torchaudio.transforms.AmplitudeToDB(stype=stype)
        amp_to_db.amin = 1e-5

        return amp_to_db(mels).clamp(min=-50, max=80)


    def get_label_from_path(self, file_path: Path) -> Optional[Tuple[str, int]]:
        parts = [p for p in file_path.parts]
        for class_name, class_id in self.class_name_to_id.items():
            if class_name in parts:
                return class_name, class_id
        return None

    def scan_files_with_labels(self, audio_exts: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")) -> None:
        collected: List[Dict[str, Any]] = []
        print(f"Scanning directory: {self.data_dir}")
        print(f"Class names: {list(self.class_name_to_id.keys())}")
        
        file_count = 0
        labeled_count = 0
        
        for path in self.data_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in audio_exts:
                file_count += 1
                label = self.get_label_from_path(path)
                if label is None:
                    print(f"No label found for: {path}")
                    continue
                label_name, label_id = label
                labeled_count += 1
                collected.append({
                    "path": path,
                    "label_name": label_name,
                    "label_id": label_id,
                })
        
        print(f"Total audio files found: {file_count}")
        print(f"Files with labels: {labeled_count}")
        self.meta = collected

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int):
        item = self.meta[index]
        path = item["path"]
        label_id = item["label_id"]

        waveform, sample_rate = self.load_audio(str(path))
        # mono 변환
        if waveform.dim() == 2:
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        # mel 스펙트로그램
        mel = self.waveform_to_log_mel(
            waveform=waveform,
            **self.mel_kwargs
        )

        if self.transform is not None:
            mel = self.transform(mel)

        y = torch.tensor(label_id, dtype=torch.long)
        return mel, y

