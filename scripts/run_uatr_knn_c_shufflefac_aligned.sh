#!/usr/bin/env bash
set -euo pipefail

# Run UATR_KNN-C on DeepShip with the same core setup used by the
# ShuffleFAC gamma=16 strict recording-level experiment:
# segment_length=3s, split=7:1:2, n_fft/window=4096, hop=2048, mel=128.
#
# Example output directory:
# results/DeepShip_0429_1530_Clean_VarC_ShuffleFACAligned_3s_7_1_2/G1_P1_TE1_TA1_Clean/Run_0
#
# Main output files per run:
# metrics.txt, classification_report.txt, confusion_matrix.png,
# model_config.json, split_audit.txt, split_audit_pre.txt,
# split_audit_post.txt, test_predictions.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CURRENT_TIME="$(date +"%m%d_%H%M")"
EXP_TIME="${EXP_TIME:-${CURRENT_TIME}_Clean_VarC_ShuffleFACAligned_3s_7_1_2}"

"${PYTHON_BIN}" demo_light.py \
  --model UATR_KNN \
  --uatr_variant C \
  --data_selection 0 \
  --split_protocol recording_level \
  --segment_length 3 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --window_length 4096 \
  --hop_length 2048 \
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
  --num_runs "${NUM_RUNS:-1}" \
  --exp_time "${EXP_TIME}"
