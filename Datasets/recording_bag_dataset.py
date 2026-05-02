"""Recording-level bag wrapper for segment datasets."""

import os
import random
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class RecordingBagDataset(Dataset):
    """Group segment samples from the same recording into a fixed-size clip bag.

    The wrapped dataset must expose `segment_list`, where each item contains at
    least `(file_path, label)`. `__getitem__` returns a dict with waveform bag
    `[S, L]` and `recording_id`, plus the recording label.
    """

    def __init__(self, base_dataset, clips_per_recording=4, random_sample=True):
        self.base_dataset = base_dataset
        self.clips_per_recording = int(clips_per_recording)
        self.random_sample = bool(random_sample)
        if self.clips_per_recording <= 0:
            raise ValueError("clips_per_recording must be positive")
        if not hasattr(base_dataset, "segment_list"):
            raise ValueError("base_dataset must expose segment_list")

        self.recording_to_indices = OrderedDict()
        self.recording_to_label = {}
        for index, item in enumerate(base_dataset.segment_list):
            file_path, label = item[0], int(item[1])
            recording_id = self._recording_id_from_path(file_path)
            self.recording_to_indices.setdefault(recording_id, []).append(index)
            old_label = self.recording_to_label.get(recording_id)
            if old_label is not None and old_label != label:
                raise ValueError(f"Recording {recording_id} has mixed labels: {old_label}, {label}")
            self.recording_to_label[recording_id] = label

        self.recording_ids = list(self.recording_to_indices.keys())

    @staticmethod
    def _recording_id_from_path(file_path):
        return os.path.normpath(os.path.dirname(os.path.abspath(str(file_path))))

    def __len__(self):
        return len(self.recording_ids)

    def _sample_indices(self, indices):
        if len(indices) >= self.clips_per_recording:
            if self.random_sample:
                return random.sample(indices, self.clips_per_recording)
            return list(indices[: self.clips_per_recording])

        if self.random_sample:
            return [random.choice(indices) for _ in range(self.clips_per_recording)]

        repeated = []
        while len(repeated) < self.clips_per_recording:
            repeated.extend(indices)
        return repeated[: self.clips_per_recording]

    @staticmethod
    def _extract_waveform(sample):
        x, _ = sample
        if isinstance(x, dict):
            waveform = x.get("waveform", x.get("x"))
            if waveform is None:
                raise ValueError("Wrapped dataset item dict must contain waveform")
            return waveform
        return x

    def __getitem__(self, idx):
        recording_id = self.recording_ids[idx]
        indices = self._sample_indices(self.recording_to_indices[recording_id])
        waveforms = []
        for segment_idx in indices:
            sample = self.base_dataset[segment_idx]
            waveform = self._extract_waveform(sample)
            waveforms.append(waveform)

        clips = torch.stack(waveforms, dim=0)
        label = torch.tensor(self.recording_to_label[recording_id], dtype=torch.long)
        return {"waveform": clips, "recording_id": recording_id}, label
