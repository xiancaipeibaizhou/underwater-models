#!/usr/bin/env bash
set -euo pipefail

# UATR_KNN-C multi-seed stability experiment.
# Fixed split: strict recording-level, 5s, 6:2:2, 2048/512, 128 mel bins.
# Training seeds are 42/43/44 via demo_light.py --num_runs 3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CURRENT_TIME="$(date +"%m%d_%H%M")"
EXP_TIME="${EXP_TIME:-${CURRENT_TIME}_Clean_VarC_multiseed_5s_6_2_2_2048_512}"

"${PYTHON_BIN}" -u demo_light.py \
  --model UATR_KNN \
  --uatr_variant C \
  --data_selection 0 \
  --split_protocol recording_level \
  --segment_length 5 \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --window_length 2048 \
  --hop_length 512 \
  --number_mels 128 \
  --sample_rate 16000 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --val_batch_size "${VAL_BATCH_SIZE:-32}" \
  --test_batch_size "${TEST_BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --num_epochs "${NUM_EPOCHS:-150}" \
  --patience "${PATIENCE:-10}" \
  --lr "${LR:-1e-3}" \
  --dropout "${DROPOUT:-0.2}" \
  --weight_decay "${WEIGHT_DECAY:-1e-5}" \
  --num_runs 3 \
  --latency_warmup "${LATENCY_WARMUP:-30}" \
  --latency_repeats "${LATENCY_REPEATS:-100}" \
  --exp_time "${EXP_TIME}"
