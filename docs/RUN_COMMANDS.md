# Run Commands

以下命令用于服务器正式实验。本地 Codex 只做语法和最小 forward 检查，不跑完整训练。

固定 split 对比时使用 `--split_seed 42 --model_seed 42`。其中 `split_seed` 只控制
train/val/test 划分，`model_seed + run_index` 控制训练随机性；因此并行跑
`--run_index 0/1/2 --num_runs 1` 时，三次训练 seed 分别是 42/43/44，但 split 保持一致。

## 当前主线：ShuffleFAC_CLIPGRAPH

当前不再继续优化 `FA_UATR_KNN_V2`，优先验证 recording-level aggregation：

1. `ShuffleFAC` segment-level 训练/测试，同时输出 recording-level voting metrics。
2. `ShuffleFAC_CLIPGRAPH` 使用同一 recording 的多个 3s clips 做图聚合。

### DeepShip ShuffleFAC 3s / 7:1:2 + Recording Voting

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model ShuffleFAC \
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
  --train_batch_size 48 \
  --val_batch_size 48 \
  --test_batch_size 48 \
  --num_workers 2 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --split_seed 42 \
  --model_seed 42 \
  --run_index 0 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_ShuffleFAC_3s_7_1_2_recording_eval" \
  > run_deepship_shufflefac_3s_recording_eval.log 2>&1 &
```

### DeepShip ShuffleFAC_CLIPGRAPH S=4

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model ShuffleFAC_CLIPGRAPH \
  --recording_bag_mode \
  --clips_per_recording 4 \
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
  --train_batch_size 8 \
  --val_batch_size 8 \
  --test_batch_size 8 \
  --num_workers 2 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --shufflefac_gamma 16 \
  --graph_hidden_dim 128 \
  --graph_layers 1 \
  --graph_k 2 \
  --edge_mode temporal_similarity \
  --graph_pooling attention \
  --split_seed 42 \
  --model_seed 42 \
  --run_index 0 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_ShuffleFAC_CLIPGRAPH_3s_7_1_2_S4_seed42" \
  > run_deepship_shufflefac_clipgraph_s4_seed42.log 2>&1 &
```

### ShipsEar ShuffleFAC_CLIPGRAPH S=4

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model ShuffleFAC_CLIPGRAPH \
  --recording_bag_mode \
  --clips_per_recording 4 \
  --data_selection 1 \
  --split_protocol recording_level \
  --segment_length 3 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --window_length 4096 \
  --hop_length 2048 \
  --number_mels 128 \
  --sample_rate 16000 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --test_batch_size 8 \
  --num_workers 2 \
  --num_epochs 200 \
  --patience 200 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --shufflefac_gamma 16 \
  --graph_hidden_dim 128 \
  --graph_layers 1 \
  --graph_k 2 \
  --edge_mode temporal_similarity \
  --graph_pooling attention \
  --split_seed 42 \
  --model_seed 42 \
  --run_index 0 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_ShipsEar_ShuffleFAC_CLIPGRAPH_3s_7_1_2_S4_seed42" \
  > run_shipsear_shufflefac_clipgraph_s4_seed42.log 2>&1 &
```

## FA_UATR_KNN_V2 历史命令

以下命令仅作为历史记录保留；当前主线不再继续投入 `FA_UATR_KNN_V2`。
若需要复现旧消融，原建议顺序为：

1. 先跑 DeepShip 5 epoch smoke test，确认数据、配置和 forward/日志都正常。
2. smoke test 正常后，再跑 DeepShip 200 epoch 正式实验。
3. DeepShip 正常后，再跑 ShipsEar 200 epoch。

Codex 本地不要执行这些训练命令；这些命令只用于服务器训练。

### DeepShip FA_UATR_KNN_V2 Smoke Test

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model FA_UATR_KNN_V2 \
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
  --num_epochs 5 \
  --patience 5 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --fusion_dim 128 \
  --knn_k 8 \
  --uatr_depth 1 \
  --fa_target_freq 4 \
  --fa_arch parallel \
  --pos_type 2d \
  --knn_metric cosine \
  --knn_source pre_trans \
  --gate_type token \
  --gate_init_bias -2.0 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_FA_UATR_KNN_V2_smoke" \
  > run_deepship_fa_uatr_knn_v2_smoke.log 2>&1 &
```

### DeepShip FA_UATR_KNN_V2 200 Epoch

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model FA_UATR_KNN_V2 \
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
  --patience 40 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --fusion_dim 128 \
  --knn_k 8 \
  --uatr_depth 1 \
  --fa_target_freq 4 \
  --fa_arch parallel \
  --pos_type 2d \
  --knn_metric cosine \
  --knn_source pre_trans \
  --gate_type token \
  --gate_init_bias -2.0 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_DeepShip_FA_UATR_KNN_V2_5s_6_2_2" \
  > run_deepship_fa_uatr_knn_v2_200epoch.log 2>&1 &
```

### ShipsEar FA_UATR_KNN_V2 200 Epoch

```bash
CURRENT_TIME=$(date +"%m%d_%H%M")
nohup python demo_light.py \
  --model FA_UATR_KNN_V2 \
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
  --patience 40 \
  --lr 1e-3 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --fusion_dim 128 \
  --knn_k 8 \
  --uatr_depth 1 \
  --fa_target_freq 4 \
  --fa_arch parallel \
  --pos_type 2d \
  --knn_metric cosine \
  --knn_source pre_trans \
  --gate_type token \
  --gate_init_bias -2.0 \
  --num_runs 1 \
  --exp_time "${CURRENT_TIME}_ShipsEar_FA_UATR_KNN_V2_5s_7_1_2" \
  > run_shipsear_fa_uatr_knn_v2_200epoch.log 2>&1 &
```

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
