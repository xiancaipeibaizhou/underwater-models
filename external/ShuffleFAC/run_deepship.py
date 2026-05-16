import argparse
import atexit
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.shuffleFAC import shuffleFAC


DATASET_CLASS_MAPPINGS = {
    "DeepShip": {"Cargo": 0, "Passengership": 1, "Tanker": 2, "Tug": 3},
    "ShipsEar": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
}

CLASS_MAPPING = DATASET_CLASS_MAPPINGS["DeepShip"]
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
DISABLE_PROGRESS = False


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"
        self.errors = getattr(streams[0], "errors", "replace") if streams else "replace"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return bool(self.streams and self.streams[0].isatty())

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def install_run_log(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    def close_log():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()

    atexit.register(close_log)
    print(f"run log: {log_path}", flush=True)
    print(f"command: {' '.join(sys.argv)}", flush=True)
    return log_path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def as_resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filters_from_gamma(gamma: int) -> list:
    return [gamma, gamma * 2, gamma * 4, gamma * 8, gamma * 8, gamma * 8, gamma * 8]


def wav_info(path: Path):
    try:
        sample_rate, signal = wavfile.read(str(path), mmap=True)
    except Exception:
        sample_rate, signal = wavfile.read(str(path), mmap=False)
    return sample_rate, int(signal.shape[0])


def read_waveform(path: Path):
    sample_rate, signal = wavfile.read(str(path), mmap=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if np.issubdtype(signal.dtype, np.integer):
        info = np.iinfo(signal.dtype)
        scale = float(max(abs(info.min), info.max))
        signal = signal.astype(np.float32) / scale
    else:
        signal = signal.astype(np.float32)

    return sample_rate, signal


def resample_waveform(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(signal, up, down).astype(np.float32, copy=False)


def discover_segments(
    data_root: Path,
    class_mapping: dict,
    segment_length: float,
    include_direct_wavs: bool = False,
    include_subdir_wavs: bool = True,
):
    data_root = data_root.resolve()
    entries = []
    recordings = {}
    skipped = []

    for class_name in sorted(class_mapping.keys()):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            skipped.append({"path": str(class_dir), "reason": "missing_class_dir"})
            continue

        if include_direct_wavs:
            for wav_path in sorted(class_dir.glob("*.wav")):
                recording_id = f"{class_name}/{wav_path.stem}"
                recordings[recording_id] = {
                    "recording_id": recording_id,
                    "class_name": class_name,
                    "label": class_mapping[class_name],
                    "paths": [str(wav_path.resolve())],
                }

                try:
                    sample_rate, n_frames = wav_info(wav_path)
                except Exception as exc:
                    skipped.append({"path": str(wav_path), "reason": f"wav_info_failed: {exc}"})
                    continue

                duration = n_frames / float(sample_rate)
                n_segments = int(duration // segment_length)
                for segment_index in range(n_segments):
                    entries.append(
                        {
                            "path": str(wav_path.resolve()),
                            "class_name": class_name,
                            "label": class_mapping[class_name],
                            "recording_id": recording_id,
                            "segment_index": segment_index,
                            "start_sec": segment_index * segment_length,
                            "duration_sec": segment_length,
                            "source_sample_rate": sample_rate,
                        }
                    )

        if not include_subdir_wavs:
            continue

        for recording_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
            wav_paths = sorted(recording_dir.glob("*.wav"))
            if not wav_paths:
                skipped.append({"path": str(recording_dir), "reason": "no_wav"})
                continue

            recording_id = f"{class_name}/{recording_dir.name}"
            recordings[recording_id] = {
                "recording_id": recording_id,
                "class_name": class_name,
                "label": class_mapping[class_name],
                "paths": [str(p.resolve()) for p in wav_paths],
            }

            for wav_path in wav_paths:
                try:
                    sample_rate, n_frames = wav_info(wav_path)
                except Exception as exc:
                    skipped.append({"path": str(wav_path), "reason": f"wav_info_failed: {exc}"})
                    continue

                duration = n_frames / float(sample_rate)
                n_segments = int(duration // segment_length)
                for segment_index in range(n_segments):
                    entries.append(
                        {
                            "path": str(wav_path.resolve()),
                            "class_name": class_name,
                            "label": class_mapping[class_name],
                            "recording_id": recording_id,
                            "segment_index": segment_index,
                            "start_sec": segment_index * segment_length,
                            "duration_sec": segment_length,
                            "source_sample_rate": sample_rate,
                        }
                    )

    return entries, recordings, skipped


def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    if n >= 3 and n_test == 0:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
    return n_train, n_val, n_test


def split_frame_level(entries, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    by_class = defaultdict(list)
    for entry in entries:
        by_class[entry["class_name"]].append(entry)

    for class_name in sorted(by_class.keys()):
        class_entries = list(by_class[class_name])
        rng.shuffle(class_entries)
        n_train, n_val, _ = split_counts(len(class_entries), train_ratio, val_ratio, test_ratio)
        splits["train"].extend(class_entries[:n_train])
        splits["val"].extend(class_entries[n_train : n_train + n_val])
        splits["test"].extend(class_entries[n_train + n_val :])

    for split_name in splits:
        rng.shuffle(splits[split_name])
    return splits


def split_recording_level(entries, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    rng = random.Random(seed)
    entries_by_recording = defaultdict(list)
    recordings_by_class = defaultdict(list)

    for entry in entries:
        entries_by_recording[entry["recording_id"]].append(entry)
    for recording_id, rec_entries in entries_by_recording.items():
        recordings_by_class[rec_entries[0]["class_name"]].append(recording_id)

    split_recordings = {"train": set(), "val": set(), "test": set()}
    for class_name in sorted(recordings_by_class.keys()):
        rec_ids = sorted(recordings_by_class[class_name])
        rng.shuffle(rec_ids)
        n_train, n_val, _ = split_counts(len(rec_ids), train_ratio, val_ratio, test_ratio)
        split_recordings["train"].update(rec_ids[:n_train])
        split_recordings["val"].update(rec_ids[n_train : n_train + n_val])
        split_recordings["test"].update(rec_ids[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for split_name, rec_ids in split_recordings.items():
        for recording_id in sorted(rec_ids):
            splits[split_name].extend(entries_by_recording[recording_id])
        rng.shuffle(splits[split_name])
    return splits


def expected_metadata(args, data_root: Path, class_mapping: dict):
    return {
        "dataset": args.dataset_name,
        "protocol": args.split_protocol,
        "random_seed": args.random_seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "parent_folder": as_resolved(data_root),
        "class_mapping": class_mapping,
        "segment_length": args.segment_length,
        "target_sample_rate": args.sample_rate,
        "include_direct_wavs": args.include_direct_wavs,
        "include_subdir_wavs": args.include_subdir_wavs,
    }


def metadata_matches(found: dict, expected: dict) -> bool:
    for key, value in expected.items():
        if found.get(key) != value:
            return False
    return True


def load_or_create_split(args, data_root: Path):
    split_json = Path(args.split_json)
    expected = expected_metadata(args, data_root, CLASS_MAPPING)

    if split_json.exists():
        with split_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if metadata_matches(payload.get("metadata", {}), expected):
            return payload, False

    entries, recordings, skipped = discover_segments(
        data_root,
        CLASS_MAPPING,
        args.segment_length,
        include_direct_wavs=args.include_direct_wavs,
        include_subdir_wavs=args.include_subdir_wavs,
    )
    if args.split_protocol == "recording_level":
        segment_lists = split_recording_level(
            entries, args.random_seed, args.train_ratio, args.val_ratio, args.test_ratio
        )
    elif args.split_protocol == "frame_level":
        segment_lists = split_frame_level(
            entries, args.random_seed, args.train_ratio, args.val_ratio, args.test_ratio
        )
    else:
        raise ValueError(f"Unsupported split protocol: {args.split_protocol}")

    payload = {
        "metadata": {
            **expected,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "shuffle": True,
            "source_repo": "https://github.com/KNU-LMAP/ShuffleFAC",
            "source_commit": args.source_commit,
            "skipped_files": skipped,
            "total_recordings": len(recordings),
            "total_segments": len(entries),
        },
        "segment_lists": segment_lists,
    }

    split_json.parent.mkdir(parents=True, exist_ok=True)
    with split_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload, True


def audit_split(segment_lists: dict, audit_path: Path, class_mapping: dict, dataset_name: str):
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "splits": {},
        "recording_overlap": {},
    }
    recording_sets = {}

    for split_name in ["train", "val", "test"]:
        entries = segment_lists[split_name]
        recording_sets[split_name] = {e["recording_id"] for e in entries}
        class_segment_counts = Counter(e["class_name"] for e in entries)
        class_recording_sets = defaultdict(set)
        for entry in entries:
            class_recording_sets[entry["class_name"]].add(entry["recording_id"])

        audit["splits"][split_name] = {
            "segments": len(entries),
            "recordings": len(recording_sets[split_name]),
            "class_distribution": {
                class_name: {
                    "segments": class_segment_counts.get(class_name, 0),
                    "recordings": len(class_recording_sets.get(class_name, set())),
                }
                for class_name in sorted(class_mapping.keys())
            },
        }

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = sorted(recording_sets[a].intersection(recording_sets[b]))
        audit["recording_overlap"][f"{a}-{b}"] = {
            "count": len(overlap),
            "recordings": overlap,
        }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        f.write(f"ShuffleFAC {audit.get('dataset', 'dataset')} split audit\n")
        f.write(f"timestamp: {audit['timestamp']}\n\n")
        for split_name in ["train", "val", "test"]:
            split = audit["splits"][split_name]
            f.write(f"[{split_name}]\n")
            f.write(f"segments: {split['segments']}\n")
            f.write(f"recordings: {split['recordings']}\n")
            for class_name, counts in split["class_distribution"].items():
                f.write(
                    f"  {class_name}: segments={counts['segments']}, "
                    f"recordings={counts['recordings']}\n"
                )
            f.write("\n")
        for pair, detail in audit["recording_overlap"].items():
            f.write(f"{pair} recording overlap: {detail['count']}\n")
            if detail["count"]:
                preview = ", ".join(detail["recordings"][:20])
                suffix = " ..." if detail["count"] > 20 else ""
                f.write(f"  {preview}{suffix}\n")
        f.write("\n")
        if any(v["count"] > 0 for v in audit["recording_overlap"].values()):
            f.write("WARNING: recording-level overlap detected.\n")
        else:
            f.write("OK: recording-level overlap = 0 for all split pairs.\n")

    return audit


def split_cache_key(entries: list, metadata: dict, feats_cfg: dict, split_name: str):
    h = hashlib.sha1()
    stable_meta = {
        "split_name": split_name,
        "metadata": {k: v for k, v in metadata.items() if k != "timestamp"},
        "feats": feats_cfg,
        "n_entries": len(entries),
    }
    h.update(json.dumps(stable_meta, sort_keys=True).encode("utf-8"))
    for entry in entries:
        h.update(
            f"{entry['path']}|{entry['segment_index']}|{entry['label']}|"
            f"{entry['duration_sec']}\n".encode("utf-8")
        )
    return h.hexdigest()


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_payload: dict, return_metadata: bool = False):
        self.features = cache_payload["features"]
        self.labels = cache_payload["labels"].long()
        self.entries = cache_payload.get("entries", [])
        self.return_metadata = return_metadata

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        if self.return_metadata:
            entry = self.entries[index] if index < len(self.entries) else {}
            return (
                self.features[index].float(),
                self.labels[index],
                entry.get("path", ""),
                entry.get("recording_id", ""),
                int(entry.get("segment_index", -1)),
            )
        return self.features[index].float(), self.labels[index]


class LogMelExtractor:
    def __init__(self, feats_cfg: dict, device: torch.device):
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=feats_cfg["sample_rate"],
            n_fft=feats_cfg["n_fft"],
            hop_length=feats_cfg["hop_length"],
            win_length=feats_cfg["win_length"],
            f_min=feats_cfg["f_min"],
            f_max=feats_cfg["f_max"],
            n_mels=feats_cfg["n_mels"],
            power=1,
            center=True,
            norm=None,
            mel_scale="htk",
            window_fn=torch.hamming_window,
        ).to(device)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="magnitude").to(device)
        self.amp_to_db.amin = 1e-5

    @torch.no_grad()
    def __call__(self, waveform_batch: torch.Tensor) -> torch.Tensor:
        waveform_batch = waveform_batch.to(self.device, non_blocking=True)
        mel = self.mel(waveform_batch)
        mel_db = self.amp_to_db(mel).clamp(min=-50, max=80)
        return mel_db.unsqueeze(1).cpu()


def chunks(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def build_feature_cache(
    split_name: str,
    entries: list,
    metadata: dict,
    feats_cfg: dict,
    cache_dir: Path,
    cache_dtype: str,
    feature_batch_size: int,
    feature_device: torch.device,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = split_cache_key(entries, metadata, feats_cfg, split_name)
    cache_path = cache_dir / f"{metadata['protocol']}_{split_name}_{cache_key[:16]}.pt"

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("cache_key") == cache_key:
            if "entries" not in payload:
                payload["entries"] = entries
                torch.save(payload, cache_path)
            return payload, cache_path, True

    dtype = torch.float16 if cache_dtype == "float16" else torch.float32
    n_frames = int((feats_cfg["sample_rate"] * metadata["segment_length"]) / feats_cfg["hop_length"]) + 1
    features = torch.empty((len(entries), 1, feats_cfg["n_mels"], n_frames), dtype=dtype)
    labels = torch.empty((len(entries),), dtype=torch.long)
    extractor = LogMelExtractor(feats_cfg, feature_device)
    seg_samples = int(feats_cfg["sample_rate"] * metadata["segment_length"])

    by_path = defaultdict(list)
    for out_index, entry in enumerate(entries):
        by_path[entry["path"]].append((out_index, entry))
        labels[out_index] = int(entry["label"])

    progress = tqdm(
        sorted(by_path.items()),
        desc=f"Cache {split_name}",
        dynamic_ncols=True,
        disable=DISABLE_PROGRESS,
    )
    for path_str, indexed_entries in progress:
        path = Path(path_str)
        sample_rate, signal = read_waveform(path)
        signal = resample_waveform(signal, sample_rate, feats_cfg["sample_rate"])
        indexed_entries = sorted(indexed_entries, key=lambda item: item[1]["segment_index"])

        for batch in chunks(indexed_entries, feature_batch_size):
            batch_waves = []
            batch_out_indices = []
            for out_index, entry in batch:
                start = int(entry["segment_index"] * seg_samples)
                end = start + seg_samples
                segment = signal[start:end]
                if segment.shape[0] < seg_samples:
                    padded = np.zeros((seg_samples,), dtype=np.float32)
                    padded[: segment.shape[0]] = segment
                    segment = padded
                batch_waves.append(segment.astype(np.float32, copy=False))
                batch_out_indices.append(out_index)
            wave_tensor = torch.from_numpy(np.stack(batch_waves, axis=0))
            mel = extractor(wave_tensor).to(dtype=dtype)
            features[torch.tensor(batch_out_indices, dtype=torch.long)] = mel

    payload = {
        "cache_key": cache_key,
        "features": features,
        "labels": labels,
        "metadata": metadata,
        "feats": feats_cfg,
        "split_name": split_name,
        "entries": entries,
    }
    torch.save(payload, cache_path)
    return payload, cache_path, False


def build_loaders(segment_lists: dict, metadata: dict, feats_cfg: dict, args, device: torch.device):
    cache_dir = Path(args.feature_cache_dir)
    feature_device = torch.device(args.feature_device)
    if feature_device.type == "cuda" and not torch.cuda.is_available():
        feature_device = torch.device("cpu")

    pin_memory = device.type == "cuda"
    if args.test_only and (args.val_cache_path or args.test_cache_path):
        loaders = {}
        cache_paths = {}
        for split_name, cache_arg in [("val", args.val_cache_path), ("test", args.test_cache_path)]:
            if not cache_arg:
                continue
            cache_path = Path(cache_arg)
            payload = torch.load(cache_path, map_location="cpu")
            dataset = CachedFeatureDataset(payload, return_metadata=True)
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            cache_paths[split_name] = str(cache_path)
        if "test" in loaders:
            return loaders, cache_paths

    datasets = {}
    cache_paths = {}
    split_names = ["test"] if args.test_only else ["train", "val", "test"]
    for split_name in split_names:
        payload, cache_path, reused = build_feature_cache(
            split_name=split_name,
            entries=segment_lists[split_name],
            metadata=metadata,
            feats_cfg=feats_cfg,
            cache_dir=cache_dir,
            cache_dtype=args.cache_dtype,
            feature_batch_size=args.feature_batch_size,
            feature_device=feature_device,
        )
        print(f"{split_name} feature cache: {cache_path} ({'reused' if reused else 'created'})", flush=True)
        datasets[split_name] = CachedFeatureDataset(payload, return_metadata=(split_name == "test"))
        cache_paths[split_name] = str(cache_path)

    loaders = {}
    if "train" in datasets:
        loaders["train"] = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    if "val" in datasets:
        loaders["val"] = DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    loaders["test"] = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return loaders, cache_paths


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_seen = 0
    for x, y in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True, disable=DISABLE_PROGRESS):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(y.size(0))
        total_seen += int(y.size(0))
    return total_loss / max(1, total_seen)


def normalize_metadata(values, n_items: int, default=""):
    if values is None:
        return [default] * n_items
    if isinstance(values, str):
        return [values]
    if torch.is_tensor(values):
        values = values.cpu().tolist()
    return [str(value) for value in values]


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_details=False):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true = []
    y_pred = []
    y_prob = []
    y_logits = []
    file_paths = []
    recording_ids = []
    segment_indices = []
    for batch in tqdm(loader, desc="Eval", leave=False, dynamic_ncols=True, disable=DISABLE_PROGRESS):
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            x, y, paths, batch_recording_ids, batch_segment_indices = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, paths = batch
            batch_recording_ids = None
            batch_segment_indices = None
        else:
            x, y = batch
            paths = None
            batch_recording_ids = None
            batch_segment_indices = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * int(y.size(0))
        total_seen += int(y.size(0))
        y_true.append(y.cpu())
        y_pred.append(pred.cpu())
        if collect_details:
            y_prob.append(probs.cpu())
            y_logits.append(logits.cpu())
            batch_size = int(y.size(0))
            file_paths.extend(normalize_metadata(paths, batch_size, ""))
            recording_ids.extend(normalize_metadata(batch_recording_ids, batch_size, ""))
            segment_indices.extend(normalize_metadata(batch_segment_indices, batch_size, "-1"))

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_seen)
    if collect_details:
        y_prob = torch.cat(y_prob).numpy()
        y_logits = torch.cat(y_logits).numpy()
        return metrics, y_true, y_pred, y_prob, y_logits, file_paths, recording_ids, segment_indices
    return metrics, y_true, y_pred


def compute_metrics(y_true, y_pred):
    labels = [CLASS_MAPPING[name] for name in sorted(CLASS_MAPPING.keys(), key=lambda n: CLASS_MAPPING[n])]
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "Precision weighted": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "Recall weighted": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def compute_recording_metrics(y_true, y_logits, recording_ids):
    metrics_by_strategy, prediction_payloads = compute_recording_aggregation_metrics(
        y_true=y_true,
        y_logits=y_logits,
        y_prob=softmax_numpy(y_logits),
        y_pred=np.argmax(y_logits, axis=1),
        recording_ids=recording_ids,
        strategies=["mean_logits"],
    )
    if "mean_logits" not in metrics_by_strategy:
        return None, [], np.array([]), np.array([]), np.array([])
    payload = prediction_payloads["mean_logits"]
    return (
        metrics_by_strategy["mean_logits"],
        payload["recording_ids"],
        payload["y_true"],
        payload["y_pred"],
        payload["scores"],
    )


def _recording_groups(y_true, y_logits, y_prob, y_pred, recording_ids):
    groups = defaultdict(lambda: {"labels": [], "logits": [], "probs": [], "preds": []})
    for target, logits, probs, pred, recording_id in zip(y_true, y_logits, y_prob, y_pred, recording_ids):
        if not recording_id:
            continue
        group = groups[recording_id]
        group["labels"].append(int(target))
        group["logits"].append(np.asarray(logits, dtype=np.float64))
        group["probs"].append(np.asarray(probs, dtype=np.float64))
        group["preds"].append(int(pred))
    return groups


def _top_fraction_indices(confidence, fraction):
    n_items = int(confidence.shape[0])
    keep = max(1, int(math.ceil(n_items * fraction)))
    return np.argsort(-confidence)[:keep]


def _trimmed_mean(values, trim_fraction=0.1):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 5:
        return values.mean(axis=0)
    trim = int(math.floor(values.shape[0] * trim_fraction))
    if trim <= 0 or values.shape[0] - 2 * trim <= 0:
        return values.mean(axis=0)
    sorted_values = np.sort(values, axis=0)
    return sorted_values[trim:-trim].mean(axis=0)


def aggregate_recording_scores(logits, probs, preds, strategy):
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.int64)
    confidence = probs.max(axis=1)

    if strategy == "mean_logits":
        return logits.mean(axis=0)
    if strategy == "mean_probs":
        return probs.mean(axis=0)
    if strategy == "majority_vote":
        counts = np.bincount(preds, minlength=probs.shape[1]).astype(np.float64)
        return counts + probs.mean(axis=0) * 1e-6
    if strategy == "topk_confident_logits_25":
        return logits[_top_fraction_indices(confidence, 0.25)].mean(axis=0)
    if strategy == "topk_confident_logits_50":
        return logits[_top_fraction_indices(confidence, 0.50)].mean(axis=0)
    if strategy == "topk_confident_logits_75":
        return logits[_top_fraction_indices(confidence, 0.75)].mean(axis=0)
    if strategy == "entropy_filtered_logits":
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        threshold = float(np.median(entropy))
        keep = np.where(entropy <= threshold)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmin(entropy))])
        return logits[keep].mean(axis=0)
    if strategy == "trimmed_mean_logits":
        return _trimmed_mean(logits)
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")


def compute_recording_aggregation_metrics(
    y_true,
    y_logits,
    y_prob,
    y_pred,
    recording_ids,
    strategies=None,
):
    strategies = strategies or [
        "mean_logits",
        "mean_probs",
        "majority_vote",
        "topk_confident_logits_25",
        "topk_confident_logits_50",
        "topk_confident_logits_75",
        "entropy_filtered_logits",
        "trimmed_mean_logits",
    ]
    groups = _recording_groups(y_true, y_logits, y_prob, y_pred, recording_ids)
    if not groups:
        return {}, {}

    ordered_recording_ids = sorted(groups.keys())
    metrics_by_strategy = {}
    prediction_payloads = {}
    for strategy in strategies:
        recording_true = []
        recording_pred = []
        recording_scores = []
        for recording_id in ordered_recording_ids:
            group = groups[recording_id]
            scores = aggregate_recording_scores(
                logits=np.stack(group["logits"], axis=0),
                probs=np.stack(group["probs"], axis=0),
                preds=np.asarray(group["preds"], dtype=np.int64),
                strategy=strategy,
            )
            recording_true.append(group["labels"][0])
            recording_scores.append(scores)
            recording_pred.append(int(np.argmax(scores)))
        recording_true = np.asarray(recording_true, dtype=np.int64)
        recording_pred = np.asarray(recording_pred, dtype=np.int64)
        recording_scores = np.stack(recording_scores, axis=0)
        metrics = compute_metrics(recording_true, recording_pred)
        metrics["Recordings"] = len(ordered_recording_ids)
        metrics_by_strategy[strategy] = metrics
        prediction_payloads[strategy] = {
            "recording_ids": ordered_recording_ids,
            "y_true": recording_true,
            "y_pred": recording_pred,
            "scores": recording_scores,
        }
    return metrics_by_strategy, prediction_payloads


def aggregation_rows(split_name: str, segment_metrics: dict, aggregation_metrics: dict):
    rows = [
        {
            "split": split_name,
            "strategy": "segment",
            "ACC": segment_metrics["ACC"],
            "Macro-F1": segment_metrics["Macro-F1"],
            "Weighted-F1": segment_metrics["Weighted-F1"],
            "Recordings": "",
        }
    ]
    for strategy, metrics in aggregation_metrics.items():
        rows.append(
            {
                "split": split_name,
                "strategy": strategy,
                "ACC": metrics["ACC"],
                "Macro-F1": metrics["Macro-F1"],
                "Weighted-F1": metrics["Weighted-F1"],
                "Recordings": metrics["Recordings"],
            }
        )
    return rows


def write_aggregation_metrics(output_dir: Path, rows: list):
    if not rows:
        return
    path = output_dir / "aggregation_metrics.csv"
    fieldnames = ["split", "strategy", "ACC", "Macro-F1", "Weighted-F1", "Recordings"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_recording_metrics_legacy(y_true, y_logits, recording_ids):
    grouped_logits = defaultdict(list)
    grouped_labels = {}
    for target, logits, recording_id in zip(y_true, y_logits, recording_ids):
        if not recording_id:
            continue
        grouped_logits[recording_id].append(logits)
        grouped_labels.setdefault(recording_id, int(target))

    if not grouped_logits:
        return None, [], np.array([]), np.array([]), np.array([])

    ordered_recording_ids = sorted(grouped_logits.keys())
    recording_logits = np.stack(
        [np.mean(np.stack(grouped_logits[recording_id], axis=0), axis=0) for recording_id in ordered_recording_ids],
        axis=0,
    )
    recording_true = np.array([grouped_labels[recording_id] for recording_id in ordered_recording_ids])
    recording_pred = np.argmax(recording_logits, axis=1)
    metrics = compute_metrics(recording_true, recording_pred)
    metrics["Recordings"] = len(ordered_recording_ids)
    return metrics, ordered_recording_ids, recording_true, recording_pred, recording_logits


def softmax_numpy(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_macs(model, input_shape):
    try:
        from thop import profile

        training_state = model.training
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(input_shape)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        model.train(training_state)
        return int(macs), "thop"
    except Exception:
        pass

    hooks = []
    macs = {"total": 0}

    def conv_hook(module, inputs, output):
        x = inputs[0]
        batch_size = int(x.shape[0])
        out = output
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        out_channels = int(out.shape[1])
        kernel_ops = int(module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups))
        macs["total"] += batch_size * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inputs, output):
        out_elements = int(output.numel())
        macs["total"] += out_elements * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    training_state = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(input_shape)
        model(dummy)
    model.train(training_state)

    for hook in hooks:
        hook.remove()
    return int(macs["total"]), "conv_linear_hooks"


def format_big_number(value: int):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


@torch.no_grad()
def benchmark_latency(model, device, input_shape, warmup=30, repeats=100):
    model.eval()
    x = torch.randn(input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def save_outputs(
    output_dir: Path,
    test_metrics: dict,
    report_text: str,
    cm,
    y_true,
    y_pred,
    y_prob,
    y_logits,
    file_paths,
    sample_recording_ids,
    sample_segment_indices,
    recording_ids,
    recording_true,
    recording_pred,
    recording_logits,
    config_payload: dict,
    class_names: list,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in test_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)

    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="ShuffleFAC Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    np.save(output_dir / "y_true.npy", y_true)
    np.save(output_dir / "y_pred.npy", y_pred)
    np.save(output_dir / "y_logits.npy", y_logits)

    pred_path = output_dir / "test_predictions.csv"
    fieldnames = [
        "sample_id",
        "file_path",
        "recording_id",
        "segment_index",
        "true_label",
        "pred_label",
        "confidence",
    ]
    fieldnames.extend([f"prob_class_{i}" for i in range(len(class_names))])
    fieldnames.extend([f"logit_class_{i}" for i in range(len(class_names))])
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, (target, pred, prob_row, logit_row) in enumerate(zip(y_true, y_pred, y_prob, y_logits)):
            row = {
                "sample_id": sample_id,
                "file_path": file_paths[sample_id] if sample_id < len(file_paths) else "",
                "recording_id": sample_recording_ids[sample_id] if sample_id < len(sample_recording_ids) else "",
                "segment_index": sample_segment_indices[sample_id] if sample_id < len(sample_segment_indices) else "",
                "true_label": int(target),
                "pred_label": int(pred),
                "confidence": float(np.max(prob_row)),
            }
            for class_idx in range(len(class_names)):
                row[f"prob_class_{class_idx}"] = float(prob_row[class_idx])
                row[f"logit_class_{class_idx}"] = float(logit_row[class_idx])
            writer.writerow(row)

    if len(recording_ids) > 0:
        recording_prob = softmax_numpy(recording_logits)
        recording_path = output_dir / "recording_predictions.csv"
        fieldnames = ["recording_id", "true_label", "pred_label", "confidence"]
        fieldnames.extend([f"prob_class_{i}" for i in range(len(class_names))])
        fieldnames.extend([f"logit_class_{i}" for i in range(len(class_names))])
        with recording_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for recording_id, target, pred, prob_row, logit_row in zip(
                recording_ids, recording_true, recording_pred, recording_prob, recording_logits
            ):
                row = {
                    "recording_id": recording_id,
                    "true_label": int(target),
                    "pred_label": int(pred),
                    "confidence": float(np.max(prob_row)),
                }
                for class_idx in range(len(class_names)):
                    row[f"prob_class_{class_idx}"] = float(prob_row[class_idx])
                    row[f"logit_class_{class_idx}"] = float(logit_row[class_idx])
                writer.writerow(row)


def append_summary_csv(output_dir: Path, test_metrics: dict):
    csv_path = output_dir.parent / "shufflefac_summary.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(test_metrics)


def parse_args():
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Run ShuffleFAC with auditable split protocols.")
    parser.add_argument("--dataset_name", choices=sorted(DATASET_CLASS_MAPPINGS.keys()), default="DeepShip")
    parser.add_argument("--data_root", default=str(root / "DeepShip"))
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "default.yaml"))
    parser.add_argument("--output_dir", default=str(root / "results" / "ShuffleFAC" / "recording_level_gamma16"))
    parser.add_argument("--split_protocol", choices=["recording_level", "frame_level"], default="recording_level")
    parser.add_argument("--source_layout", choices=["subdirs", "direct_wavs"], default=None)
    parser.add_argument("--split_json", default=str(root / "deepship_shufflefac_recording_split.json"))
    parser.add_argument("--audit_file", default=str(root / "split_audit_shufflefac_recording.txt"))
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_length", type=float, default=3.0)
    parser.add_argument("--n_fft", type=int, default=None)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--hop_length", type=int, default=None)
    parser.add_argument("--n_mels", type=int, default=None)
    parser.add_argument("--f_min", type=float, default=None)
    parser.add_argument("--f_max", type=float, default=None)
    parser.add_argument("--gamma", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--feature_cache_dir", default=str(root / "shufflefac_feature_cache"))
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--feature_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--val_cache_path", default=None)
    parser.add_argument("--test_cache_path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--allow_recording_overlap", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--source_commit", default="473c1851f20c57d39fac815a6adfa1d70c8e6ae8")
    return parser.parse_args()


def main():
    global CLASS_MAPPING, INV_CLASS_MAPPING
    global DISABLE_PROGRESS
    args = parse_args()
    DISABLE_PROGRESS = bool(args.no_progress)
    CLASS_MAPPING = DATASET_CLASS_MAPPINGS[args.dataset_name]
    INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
    if args.source_layout is None:
        args.source_layout = "direct_wavs" if args.dataset_name == "ShipsEar" else "subdirs"
    args.include_direct_wavs = args.source_layout == "direct_wavs"
    args.include_subdir_wavs = args.source_layout == "subdirs"
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    seed = args.training_seed if args.training_seed is not None else args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_run_log(output_dir)
    split_payload, created_split = load_or_create_split(args, data_root)
    print(f"split json: {args.split_json} ({'created' if created_split else 'reused'})", flush=True)

    audit = audit_split(split_payload["segment_lists"], Path(args.audit_file), CLASS_MAPPING, args.dataset_name)
    print(f"audit file: {args.audit_file}", flush=True)
    overlap_counts = {k: v["count"] for k, v in audit["recording_overlap"].items()}
    if any(count > 0 for count in overlap_counts.values()) and not args.allow_recording_overlap:
        raise RuntimeError(f"Recording overlap detected: {overlap_counts}")

    if args.prepare_only:
        return

    configs = load_yaml(Path(args.config))
    cnn_cfg = dict(configs["CNN"])
    feats_cfg = dict(configs["feats"])
    feats_cfg["sample_rate"] = args.sample_rate
    for arg_name in ["n_fft", "win_length", "hop_length", "n_mels", "f_min", "f_max"]:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            feats_cfg[arg_name] = arg_value
    cnn_cfg["n_class"] = len(CLASS_MAPPING)
    cnn_cfg["n_input_ch"] = 1
    cnn_cfg["nb_filters"] = filters_from_gamma(args.gamma)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    loaders, cache_paths = build_loaders(split_payload["segment_lists"], split_payload["metadata"], feats_cfg, args, device)

    model = shuffleFAC(**cnn_cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    input_shape = (1, cnn_cfg["n_input_ch"], feats_cfg["n_mels"], int((args.sample_rate * args.segment_length) / feats_cfg["hop_length"]) + 1)
    macs, macs_method = count_macs(model.cpu(), input_shape)
    model = model.to(device)
    latency_s = benchmark_latency(model, device, input_shape)

    criterion = nn.CrossEntropyLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    train_log_path = output_dir / "train_log.txt"

    best_val_macro = -1.0
    if args.test_only:
        best_path = Path(args.ckpt_path) if args.ckpt_path else best_path
        if not best_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for test_only: {best_path}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        epochs_without_improvement = 0
        with train_log_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_acc,val_macro_f1,val_weighted_f1\n")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            val_metrics, _, _ = evaluate(model, loaders["val"], criterion, device)
            line = (
                f"{epoch},{train_loss:.6f},{val_metrics['loss']:.6f},"
                f"{val_metrics['ACC']:.6f},{val_metrics['Macro-F1']:.6f},"
                f"{val_metrics['Weighted-F1']:.6f}"
            )
            print(line, flush=True)
            with train_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            if val_metrics["Macro-F1"] > best_val_macro:
                best_val_macro = val_metrics["Macro-F1"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_val_macro_f1": best_val_macro,
                        "cnn_cfg": cnn_cfg,
                        "feats_cfg": feats_cfg,
                        "args": vars(args),
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    print(
                        f"Early stopping at epoch {epoch}: "
                        f"no validation Macro-F1 improvement for {args.patience} epochs.",
                        flush=True,
                    )
                    break

    checkpoint = torch.load(best_path, map_location=device)
    best_val_macro = float(checkpoint.get("best_val_macro_f1", best_val_macro))
    model.load_state_dict(checkpoint["model_state"])
    aggregation_metric_rows = []
    if "val" in loaders:
        (
            val_metrics,
            val_true,
            val_pred,
            val_prob,
            val_logits,
            _val_paths,
            val_recording_ids,
            _val_segment_indices,
        ) = evaluate(
            model, loaders["val"], criterion, device, collect_details=True
        )
        val_aggregation_metrics, _ = compute_recording_aggregation_metrics(
            y_true=val_true,
            y_logits=val_logits,
            y_prob=val_prob,
            y_pred=val_pred,
            recording_ids=val_recording_ids,
        )
        aggregation_metric_rows.extend(aggregation_rows("val", val_metrics, val_aggregation_metrics))

    (
        test_metrics,
        y_true,
        y_pred,
        y_prob,
        y_logits,
        test_paths,
        test_recording_ids,
        test_segment_indices,
    ) = evaluate(
        model, loaders["test"], criterion, device, collect_details=True
    )
    test_aggregation_metrics, _ = compute_recording_aggregation_metrics(
        y_true=y_true,
        y_logits=y_logits,
        y_prob=y_prob,
        y_pred=y_pred,
        recording_ids=test_recording_ids,
    )
    aggregation_metric_rows.extend(aggregation_rows("test", test_metrics, test_aggregation_metrics))
    recording_metrics, recording_ids, recording_true, recording_pred, recording_logits = compute_recording_metrics(
        y_true, y_logits, test_recording_ids
    )
    class_names = [INV_CLASS_MAPPING[i] for i in sorted(INV_CLASS_MAPPING.keys())]
    labels = [CLASS_MAPPING[name] for name in class_names]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=6,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    final_metrics = {
        "Model": f"ShuffleFAC gamma={args.gamma}",
        "Split Protocol": args.split_protocol,
        "Recording Overlap?": json.dumps(overlap_counts),
        "ACC": test_metrics["ACC"],
        "Macro-F1": test_metrics["Macro-F1"],
        "Weighted-F1": test_metrics["Weighted-F1"],
        "Precision weighted": test_metrics["Precision weighted"],
        "Recall weighted": test_metrics["Recall weighted"],
        "Test loss": test_metrics["loss"],
        "Recording ACC": recording_metrics["ACC"] if recording_metrics else None,
        "Recording Macro-F1": recording_metrics["Macro-F1"] if recording_metrics else None,
        "Recording Weighted-F1": recording_metrics["Weighted-F1"] if recording_metrics else None,
        "Recording Count": recording_metrics["Recordings"] if recording_metrics else 0,
        "Best val Macro-F1": best_val_macro,
        "Best epoch": int(checkpoint.get("epoch", -1)),
        "Params": total_params,
        "Trainable Params": trainable_params,
        "MACs": macs,
        "MACs formatted": format_big_number(macs),
        "MACs method": macs_method,
        "Latency seconds/batch1": latency_s,
        "Latency ms/batch1": latency_s * 1000.0,
        "Seed": seed,
    }

    config_payload = {
        "official_repo": "https://github.com/KNU-LMAP/ShuffleFAC",
        "official_commit": args.source_commit,
        "official_default_yaml": configs,
        "cnn_cfg_used": cnn_cfg,
        "feats_cfg_used": feats_cfg,
        "split_metadata": split_payload["metadata"],
        "cache_paths": cache_paths,
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "Adam",
            "scheduler": None,
            "augmentation": None,
            "random_seed": args.random_seed,
            "training_seed": seed,
            "test_only": args.test_only,
            "ckpt_path": str(best_path),
            "val_cache_path": args.val_cache_path,
            "test_cache_path": args.test_cache_path,
        },
        "input_shape": input_shape,
        "macs_method": macs_method,
    }

    save_outputs(
        output_dir,
        final_metrics,
        report,
        cm,
        y_true,
        y_pred,
        y_prob,
        y_logits,
        test_paths,
        test_recording_ids,
        test_segment_indices,
        recording_ids,
        recording_true,
        recording_pred,
        recording_logits,
        config_payload,
        class_names,
    )
    write_aggregation_metrics(output_dir, aggregation_metric_rows)
    append_summary_csv(output_dir, final_metrics)
    audit_src = Path(args.audit_file)
    audit_dst = output_dir / audit_src.name
    if audit_src.resolve() != audit_dst.resolve():
        shutil.copyfile(audit_src, audit_dst)

    print(report, flush=True)
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
