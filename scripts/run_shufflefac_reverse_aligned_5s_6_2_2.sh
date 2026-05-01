#!/usr/bin/env bash
set -euo pipefail

# Reverse alignment experiment:
# ShuffleFAC gamma=16 on DeepShip using the UATR_KNN-C main setup.
#
# segment_length=5s, split=6:2:2, n_fft/window=2048, hop=512, mel=128,
# sample_rate=16000, strict recording-level split.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CURRENT_TIME="$(date +"%m%d_%H%M")"
OUT_DIR="${OUT_DIR:-results/ShuffleFAC/${CURRENT_TIME}_ShuffleFAC_gamma16_recording_5s_6_2_2_2048_512}"

"${PYTHON_BIN}" -u external/ShuffleFAC/run_deepship.py \
  --data_root DeepShip \
  --config external/ShuffleFAC/default.yaml \
  --output_dir "${OUT_DIR}" \
  --split_protocol recording_level \
  --split_json deepship_shufflefac_recording_split_5s_6_2_2.json \
  --audit_file split_audit_shufflefac_5s_6_2_2.txt \
  --random_seed 42 \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --sample_rate 16000 \
  --segment_length 5 \
  --n_fft 2048 \
  --win_length 2048 \
  --hop_length 512 \
  --n_mels 128 \
  --gamma 16 \
  --epochs "${EPOCHS:-200}" \
  --batch_size "${BATCH_SIZE:-48}" \
  --lr "${LR:-1e-3}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --feature_cache_dir shufflefac_feature_cache \
  --feature_batch_size "${FEATURE_BATCH_SIZE:-128}" \
  --feature_device "${FEATURE_DEVICE:-cuda}" \
  --device "${DEVICE:-cuda}"
