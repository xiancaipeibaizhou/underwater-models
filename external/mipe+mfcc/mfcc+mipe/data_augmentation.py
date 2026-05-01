import argparse
import csv
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np

# Compatibility for older librosa releases on newer numpy releases.
if "complex" not in np.__dict__:
    np.complex = np.complex128  # type: ignore[attr-defined]
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]

import librosa
from scipy.signal import butter, filtfilt, sosfilt
from tqdm import tqdm

from mipe_core import multi_scale_mipe

warnings.filterwarnings("ignore")
logging.getLogger("paramiko").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"


CLS_MAP = {"ClassA": 0, "ClassB": 1, "ClassC": 2, "ClassD": 3, "ClassE": 4}
CLASS_ALIASES = {
    "ClassA": ("ClassA", "A"),
    "ClassB": ("ClassB", "B"),
    "ClassC": ("ClassC", "C"),
    "ClassD": ("ClassD", "D"),
    "ClassE": ("ClassE", "E"),
}
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LEN = 512
TARGET_FRAMES = 100
MIPE_SCALES = 10
AUGMENTATIONS = [
    "time_stretch",
    "pitch_shift",
    "add_noise",
    "background_noise",
    "band_mask",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MFCC-delta and MIPE features from ShipsEar audio."
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to ShipsEar root. Can also be provided by DATA_ROOT.",
    )
    parser.add_argument(
        "--out_dir",
        default="./outputs/mipe_mfcc",
        help="Directory for npy files, manifests, and logs.",
    )
    parser.add_argument("--seg_sec", type=float, default=5.0, help="Segment length in seconds.")
    parser.add_argument(
        "--augment_per_seg",
        type=int,
        default=5,
        help="Number of augmented samples generated for each original segment.",
    )
    parser.add_argument(
        "--max_files_per_class",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Uses sorted wav files per class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        parser.error("Missing --data_root and DATA_ROOT is not set.")
    args.data_root = Path(data_root).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()

    if not args.data_root.is_dir():
        parser.error(f"ShipsEar data_root does not exist or is not a directory: {args.data_root}")
    if args.seg_sec <= 0:
        parser.error("--seg_sec must be > 0.")
    if args.augment_per_seg < 0:
        parser.error("--augment_per_seg must be >= 0.")
    if args.max_files_per_class is not None and args.max_files_per_class <= 0:
        parser.error("--max_files_per_class must be > 0 when provided.")
    return args


def setup_logging(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mipe_mfcc_feature_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(out_dir / "feature_extraction.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def fix_audio_length(y, target_len):
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


def time_stretch(y, rate=0.9):
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y, rng, noise_level=0.005):
    return y + rng.normal(0.0, noise_level, size=len(y))


def background_noise(y, rng, sr=16000, snr_db=15):
    noise = rng.normal(0.0, 1.0, size=len(y))
    b, a = butter(4, 0.1, "low", fs=sr)
    noise = filtfilt(b, a, noise)
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise**2) or 1e-10
    noise = noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return y + noise


def band_mask(y, sr, fmin=200, fmax=4000):
    nyquist = sr / 2.0
    low = max(1.0, min(float(fmin), nyquist - 2.0))
    high = max(low + 1.0, min(float(fmax), nyquist - 1.0))
    if high >= nyquist or low >= high:
        return y
    sos = butter(4, [low, high], "bandstop", fs=sr, output="sos")
    return sosfilt(sos, y)


def extract_mfcc_delta(y, sample_rate):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
    )
    if mfcc.shape[1] >= 3:
        width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
        delta = librosa.feature.delta(mfcc, width=max(3, width))
    else:
        delta = np.zeros_like(mfcc)

    mfcc = mfcc.T[:TARGET_FRAMES]
    delta = delta.T[:TARGET_FRAMES]

    if len(mfcc) < TARGET_FRAMES:
        pad = TARGET_FRAMES - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
        delta = np.pad(delta, ((0, pad), (0, 0)))
    return np.stack([mfcc, delta], axis=-1).astype(np.float32)


def extract_mipe_sequence(y):
    seq = multi_scale_mipe(np.asarray(y, dtype=np.float32), scales=MIPE_SCALES)
    return np.asarray(seq, dtype=np.float32).reshape(MIPE_SCALES)


def split_audio(y, sample_rate, seg_sec):
    win = int(round(seg_sec * sample_rate))
    if win <= 0:
        raise ValueError("Segment length is zero. Increase --seg_sec.")
    if len(y) == 0:
        return []

    segs = []
    for start in range(0, len(y), win):
        seg = y[start : start + win]
        segs.append(fix_audio_length(seg, win))
    return segs


def get_files(data_root, max_files_per_class):
    items = []
    group_lookup = {}
    group_id = 0
    for cls_name, label in CLS_MAP.items():
        wav_entries = []
        for folder_name in CLASS_ALIASES[cls_name]:
            folder = data_root / folder_name
            if folder.is_dir():
                wavs = sorted(
                    p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"
                )
                for wav_path in wavs:
                    rel_path = wav_path.relative_to(folder)
                    if len(rel_path.parts) > 1:
                        recording_name = rel_path.parts[0]
                    else:
                        recording_name = wav_path.stem
                    group_key = f"{cls_name}/{recording_name}"
                    wav_entries.append((wav_path, group_key))
        wav_entries = sorted(wav_entries, key=lambda item: str(item[0]))
        if max_files_per_class is not None:
            wav_entries = wav_entries[:max_files_per_class]
        for wav_path, group_key in wav_entries:
            if group_key not in group_lookup:
                group_lookup[group_key] = group_id
                group_id += 1
            items.append(
                {
                    "path": wav_path,
                    "label": label,
                    "class_name": cls_name,
                    "group": group_lookup[group_key],
                    "recording_key": group_key,
                }
            )
    return items


def apply_augmentation(seg, sample_rate, rng):
    aug = rng.choice(AUGMENTATIONS)
    if aug == "time_stretch":
        y = time_stretch(seg, rate=float(rng.uniform(0.8, 1.2)))
    elif aug == "pitch_shift":
        y = pitch_shift(seg, sample_rate, n_steps=int(rng.integers(-3, 4)))
    elif aug == "add_noise":
        y = add_noise(seg, rng, noise_level=float(rng.uniform(0.001, 0.01)))
    elif aug == "background_noise":
        y = background_noise(seg, rng, sample_rate, snr_db=int(rng.integers(10, 25)))
    elif aug == "band_mask":
        y = band_mask(
            seg,
            sample_rate,
            fmin=int(rng.integers(100, 500)),
            fmax=int(rng.integers(2000, 5000)),
        )
    else:
        y = seg.copy()
    return fix_audio_length(np.asarray(y, dtype=np.float32), len(seg)), aug


def process_file(item, args, rng):
    y, _ = librosa.load(item["path"], sr=SAMPLE_RATE, mono=True)
    y = np.asarray(y, dtype=np.float32)
    segs = split_audio(y, SAMPLE_RATE, args.seg_sec)

    samples = []
    for seg_idx, seg in enumerate(segs):
        sample_base = {
            "source_path": str(item["path"]),
            "class_name": item["class_name"],
            "label": item["label"],
            "group": item["group"],
            "recording_key": item["recording_key"],
            "segment_index": seg_idx,
        }

        samples.append(
            {
                **sample_base,
                "augment_name": "original",
                "mfcc": extract_mfcc_delta(seg, SAMPLE_RATE),
                "mipe": extract_mipe_sequence(seg),
            }
        )

        for _ in range(args.augment_per_seg):
            aug_y, aug_name = apply_augmentation(seg, SAMPLE_RATE, rng)
            samples.append(
                {
                    **sample_base,
                    "augment_name": aug_name,
                    "mfcc": extract_mfcc_delta(aug_y, SAMPLE_RATE),
                    "mipe": extract_mipe_sequence(aug_y),
                }
            )
    return samples


def write_manifest(out_dir, rows):
    fieldnames = [
        "sample_index",
        "source_path",
        "class_name",
        "label",
        "group",
        "recording_key",
        "segment_index",
        "augment_name",
    ]
    with open(out_dir / "sample_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "sample_index": idx,
                    "source_path": row["source_path"],
                    "class_name": row["class_name"],
                    "label": row["label"],
                    "group": row["group"],
                    "recording_key": row["recording_key"],
                    "segment_index": row["segment_index"],
                    "augment_name": row["augment_name"],
                }
            )


def main():
    args = parse_args()
    logger = setup_logging(args.out_dir)
    rng = np.random.default_rng(args.seed)

    logger.info("data_root: %s", args.data_root)
    logger.info("out_dir: %s", args.out_dir)
    logger.info(
        "seg_sec=%s augment_per_seg=%s max_files_per_class=%s seed=%s",
        args.seg_sec,
        args.augment_per_seg,
        args.max_files_per_class,
        args.seed,
    )

    items = get_files(args.data_root, args.max_files_per_class)
    if not items:
        raise SystemExit(
            f"No wav files found under {args.data_root}. Expected folders: {list(CLS_MAP)}"
        )

    class_counts = {}
    for item in items:
        class_counts[item["class_name"]] = class_counts.get(item["class_name"], 0) + 1
    logger.info("Found %d recordings: %s", len(items), class_counts)

    all_samples = []
    for item in tqdm(items, total=len(items), desc="Extracting features"):
        try:
            samples = process_file(item, args, rng)
        except Exception as exc:
            logger.exception("Failed to process %s", item["path"])
            raise SystemExit(f"Failed to process {item['path']}: {exc}") from exc
        all_samples.extend(samples)

    if not all_samples:
        raise SystemExit("No samples were generated. Check audio duration and --seg_sec.")

    mfcc = np.stack([row["mfcc"] for row in all_samples]).astype(np.float32)
    mipe = np.stack([row["mipe"] for row in all_samples]).astype(np.float32)
    labels = np.asarray([row["label"] for row in all_samples], dtype=np.int64)
    groups = np.asarray([row["group"] for row in all_samples], dtype=np.int64)
    segment_ids = np.asarray([row["segment_index"] for row in all_samples], dtype=np.int64)

    np.save(args.out_dir / "mfcc_augmented.npy", mfcc)
    np.save(args.out_dir / "mipe_augmented.npy", mipe)
    np.save(args.out_dir / "labels_augmented.npy", labels)
    np.save(args.out_dir / "groups.npy", groups)
    np.save(args.out_dir / "segment_ids.npy", segment_ids)
    write_manifest(args.out_dir, all_samples)

    config = {
        "data_root": str(args.data_root),
        "sample_rate": SAMPLE_RATE,
        "seg_sec": args.seg_sec,
        "augment_per_seg": args.augment_per_seg,
        "max_files_per_class": args.max_files_per_class,
        "seed": args.seed,
        "class_map": CLS_MAP,
        "group_policy": (
            "recording-level: immediate wav files are one group each; recursive "
            "A/recording_dir/*.wav or ClassA/recording_dir/*.wav segment files "
            "share one group id per recording_dir"
        ),
        "n_recordings": len(items),
        "n_samples": int(len(labels)),
        "mfcc_shape": list(mfcc.shape),
        "mipe_shape": list(mipe.shape),
    }
    with open(args.out_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("Saved mfcc_augmented.npy: %s", mfcc.shape)
    logger.info("Saved mipe_augmented.npy: %s", mipe.shape)
    logger.info("Saved labels_augmented.npy: %s", labels.shape)
    logger.info("Saved groups.npy: %s unique groups=%d", groups.shape, len(np.unique(groups)))
    logger.info("Done.")


if __name__ == "__main__":
    main()
