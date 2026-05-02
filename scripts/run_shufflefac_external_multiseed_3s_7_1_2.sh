#!/usr/bin/env bash
set -euo pipefail

# External ShuffleFAC gamma=16 multi-seed experiment on DeepShip.
# Fixed split seed is 42. Training seeds are 42/43/44.
# Setting: strict recording-level, 3s, 7:1:2, 4096/2048, 128 mel bins.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CURRENT_TIME="$(date +"%m%d_%H%M")"
EXP_ROOT="${EXP_ROOT:-results/ShuffleFAC/${CURRENT_TIME}_ShuffleFAC_external_gamma16_multiseed_3s_7_1_2}"

for SEED in 42 43 44; do
  "${PYTHON_BIN}" -u external/ShuffleFAC/run_deepship.py \
    --data_root DeepShip \
    --config external/ShuffleFAC/default.yaml \
    --output_dir "${EXP_ROOT}/seed_${SEED}" \
    --split_protocol recording_level \
    --split_json deepship_shufflefac_recording_split_3s_7_1_2.json \
    --audit_file "${EXP_ROOT}/seed_${SEED}/split_audit_shufflefac_3s_7_1_2.txt" \
    --random_seed 42 \
    --training_seed "${SEED}" \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --sample_rate 16000 \
    --segment_length 3 \
    --n_fft 4096 \
    --win_length 4096 \
    --hop_length 2048 \
    --n_mels 128 \
    --gamma 16 \
    --epochs "${EPOCHS:-200}" \
    --patience "${PATIENCE:-40}" \
    --batch_size "${BATCH_SIZE:-48}" \
    --lr "${LR:-1e-3}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --feature_cache_dir shufflefac_feature_cache \
    --feature_batch_size "${FEATURE_BATCH_SIZE:-128}" \
    --feature_device "${FEATURE_DEVICE:-cuda}" \
    --device "${DEVICE:-cuda}" \
    --no_progress
done
