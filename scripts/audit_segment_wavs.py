#!/usr/bin/env python3
import argparse
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def wav_info(path: Path):
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnframes()


def infer_recording_id(root: Path, wav_path: Path):
    rel = wav_path.resolve().relative_to(root.resolve())
    parts = rel.parts
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 2:
        return f"{parts[0]}/{Path(parts[1]).stem}"
    return wav_path.stem


def audit_root(root: Path, limit: int, expected_sr: int, expected_samples: Optional[int], tolerance: int):
    print(f"root: {root}")
    if not root.exists():
        print("status: MISSING")
        print()
        return

    wavs = sorted(root.rglob("*.wav"))
    checked = wavs[:limit]
    print(f"total_wavs: {len(wavs)}")
    print(f"checked_wavs: {len(checked)}")

    sample_rates = []
    lengths = []
    failures = []
    recording_groups = defaultdict(list)
    examples = []
    for wav_path in checked:
        try:
            sample_rate, n_frames = wav_info(wav_path)
        except Exception as exc:
            failures.append((wav_path, str(exc)))
            continue
        sample_rates.append(sample_rate)
        lengths.append(n_frames)
        recording_id = infer_recording_id(root, wav_path)
        recording_groups[recording_id].append(wav_path)
        if len(examples) < 5:
            examples.append((wav_path, recording_id, sample_rate, n_frames))

    print(f"sample_rates: {dict(Counter(sample_rates))}")
    if lengths:
        print(f"length_min: {min(lengths)}")
        print(f"length_median: {statistics.median(lengths)}")
        print(f"length_max: {max(lengths)}")
    if expected_samples is not None:
        bad = [
            length
            for length in lengths
            if abs(length - expected_samples) > tolerance
        ]
        print(f"expected_samples: {expected_samples}")
        print(f"bad_length_count: {len(bad)}")
    if expected_sr:
        bad_sr = [sample_rate for sample_rate in sample_rates if sample_rate != expected_sr]
        print(f"expected_sample_rate: {expected_sr}")
        print(f"bad_sample_rate_count: {len(bad_sr)}")
    if failures:
        print("read_failures:")
        for path, reason in failures[:5]:
            print(f"  {path}: {reason}")

    print("recording_id_examples:")
    for wav_path, recording_id, sample_rate, n_frames in examples:
        print(f"  {wav_path} -> {recording_id} sr={sample_rate} frames={n_frames}")

    multi = [(rid, paths) for rid, paths in recording_groups.items() if len(paths) > 1]
    if multi:
        rid, paths = multi[0]
        print(f"multi_clip_recording_example: {rid} clips={len(paths)}")
        for path in paths[:5]:
            print(f"  {path.name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Audit wav segment length and recording-id grouping.")
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--expected-sr", type=int, default=16000)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--tolerance", type=int, default=8)
    args = parser.parse_args()

    for root in args.roots:
        audit_root(Path(root), args.limit, args.expected_sr, args.expected_samples, args.tolerance)


if __name__ == "__main__":
    main()
