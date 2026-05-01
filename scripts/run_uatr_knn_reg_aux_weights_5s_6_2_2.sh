#!/usr/bin/env bash
set -euo pipefail

CURRENT_TIME=$(date +"%m%d_%H%M")
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.05}"
AUX_TAG="${AUX_LOSS_WEIGHT/./p}"

python demo_light.py \
  --model UATR_KNN_REG \
  --data_selection 0 \
  --split_protocol recording_level \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_epochs 150 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --dropout 0.2 \
  --patience 10 \
  --segment_length 5 \
  --n_mfcc 20 \
  --stft_bins 64 \
  --aux_target_dim 208 \
  --aux_loss_weight "${AUX_LOSS_WEIGHT}" \
  --num_runs 3 \
  --exp_time "${CURRENT_TIME}_Clean_UATR_KNN_REG_aux${AUX_TAG}_recording"
