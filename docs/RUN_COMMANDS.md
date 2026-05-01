# Run Commands

以下命令用于服务器正式实验。本地 Codex 只做语法和最小 forward 检查，不跑完整训练。

## DeepShip UATR_KNN-C

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
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
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_workers 8 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_UATR_KNN_C_5s_6_2_2" \
  > run_deepship_uatr_knn_c.log 2>&1 &
```

## DeepShip ShuffleFAC

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model ShuffleFAC \
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
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_workers 8 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_ShuffleFAC_gamma16_5s_6_2_2" \
  > run_deepship_shufflefac.log 2>&1 &
```

## ShipsEar UATR_KNN-C

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model UATR_KNN \
  --uatr_variant C \
  --data_selection 1 \
  --split_protocol recording_level \
  --segment_length 5 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --window_length 2048 \
  --hop_length 512 \
  --number_mels 128 \
  --sample_rate 16000 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_workers 8 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_ShipsEar_UATR_KNN_C_5s_7_1_2" \
  > run_shipsear_uatr_knn_c_7_1_2.log 2>&1 &
```

## ShipsEar ShuffleFAC

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model ShuffleFAC \
  --data_selection 1 \
  --split_protocol recording_level \
  --segment_length 5 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --window_length 2048 \
  --hop_length 512 \
  --number_mels 128 \
  --sample_rate 16000 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_workers 8 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_ShipsEar_ShuffleFAC_gamma16_5s_7_1_2" \
  > run_shipsear_shufflefac_7_1_2.log 2>&1 &
```

## 后续 DeepShip FA_UATR_KNN

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model FA_UATR_KNN \
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
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_workers 8 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --fusion_dim 128 \
  --knn_k 8 \
  --uatr_depth 1 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_FA_UATR_KNN_5s_6_2_2" \
  > run_deepship_fa_uatr_knn.log 2>&1 &
```

## 后续 ShipsEar FA_UATR_KNN

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model FA_UATR_KNN \
  --data_selection 1 \
  --split_protocol recording_level \
  --segment_length 5 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --window_length 2048 \
  --hop_length 512 \
  --number_mels 128 \
  --sample_rate 16000 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --test_batch_size 32 \
  --num_workers 8 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --fusion_dim 128 \
  --knn_k 8 \
  --uatr_depth 1 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_ShipsEar_FA_UATR_KNN_5s_7_1_2" \
  > run_shipsear_fa_uatr_knn.log 2>&1 &
```
