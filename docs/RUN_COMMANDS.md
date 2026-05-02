# Run Commands

以下命令用于服务器正式实验。本地 Codex 只做语法和最小 forward 检查，不跑完整训练。

固定 split 对比时使用 `--split_seed 42 --model_seed 42`。其中 `split_seed` 只控制
train/val/test 划分，`model_seed + run_index` 控制训练随机性；因此并行跑
`--run_index 0/1/2 --num_runs 1` 时，三次训练 seed 分别是 42/43/44，但 split 保持一致。

## 当前主线：external ShuffleFAC + recording-level aggregation

当前正式路线使用 `external/ShuffleFAC` 下的 ShuffleFAC native protocol：

```text
3s / 7:1:2 / strict recording-level split / 4096-2048 / 128 Mel / 16 kHz
```

当前主结果：

- DeepShip：`ShuffleFAC` ordinary recording-level mean-logit voting。
- ShipsEar：pretrained `ShuffleFAC` encoder + frozen deterministic multi-sample `AttentionHead`。
- GNN 方向保留为消融：`GraphHead` 和 `Graph-aware AttentionHead`。

不要把 from-scratch `ShuffleFAC_CLIPGRAPH`、`FA_UATR_KNN_V2` 或 5s demo_light 旧基线写成当前主线。

### DeepShip current result path

DeepShip 3-seed 已完成，当前主结果来自 ordinary recording-level mean-logit voting：

| Seed | Checkpoint |
| ---: | --- |
| 42 | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_42/best.pt` |
| 43 | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_43/best.pt` |
| 44 | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_44_parallel/best.pt` |

DeepShip 目前不需要用 learned head 替代 ordinary voting。若需要复核 GNN 消融，可运行 Graph-aware AttentionHead：

```bash
for item in \
  "42 seed_42" \
  "43 seed_43" \
  "44 seed_44_parallel"
do
  set -- $item
  SEED=$1
  CKPT_DIR=$2
  python -u external/ShuffleFAC/run_graphhead.py \
    --encoder_ckpt results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/${CKPT_DIR}/best.pt \
    --model_config results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/${CKPT_DIR}/model_config.json \
    --head_type graph_aware_attention \
    --edge_mode temporal_similarity \
    --output_dir results/ShuffleFAC_GRAPHHEAD/DeepShip_seed${SEED}_graph_aware_attention_S8_ms5 \
    --clips_per_recording 8 \
    --batch_size 8 \
    --epochs 50 \
    --patience 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.2 \
    --eval_samples 5 \
    --seed ${SEED}
done
```

### ShipsEar current result path

ShipsEar 当前主结果是 frozen encoder + deterministic multi-sample `AttentionHead`：

```bash
for SEED in 42 43 44
do
  python -u external/ShuffleFAC/run_graphhead.py \
    --encoder_ckpt results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_${SEED}/best.pt \
    --model_config results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_${SEED}/model_config.json \
    --head_type attention \
    --output_dir results/ShuffleFAC_GRAPHHEAD/ShipsEar_seed${SEED}_attention_S8_ms5 \
    --clips_per_recording 8 \
    --batch_size 8 \
    --epochs 50 \
    --patience 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.2 \
    --eval_samples 5 \
    --seed ${SEED}
done
```

ShipsEar GNN-guided ablation 使用 `graph_aware_attention`：

```bash
for SEED in 42 43 44
do
  python -u external/ShuffleFAC/run_graphhead.py \
    --encoder_ckpt results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_${SEED}/best.pt \
    --model_config results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_${SEED}/model_config.json \
    --head_type graph_aware_attention \
    --edge_mode temporal_similarity \
    --output_dir results/ShuffleFAC_GRAPHHEAD/ShipsEar_seed${SEED}_graph_aware_attention_S8_ms5 \
    --clips_per_recording 8 \
    --batch_size 8 \
    --epochs 50 \
    --patience 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.2 \
    --eval_samples 5 \
    --seed ${SEED}
done
```

### Frozen head settings

`run_graphhead.py` 默认遵循当前主线要求：

- 加载 external ShuffleFAC `best.pt`。
- 冻结 ShuffleFAC encoder，只训练 recording-level head。
- train 随机采样 `clips_per_recording=8`。
- val/test 使用 deterministic temporal phase sampling。
- `--eval_samples 5` 时，对同一 recording 做 5 个确定性 bags，平均 logits 后计算 recording metrics。
- 输出 `attn_entropy`、`graph_delta_norm`、`graph_res_scale`、trainable/frozen params。

## 暂停路线：FA_UATR_KNN_V2 历史命令

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

## 历史基线命令（非当前主线）

以下 5s `demo_light.py` 命令仅用于复现旧基线或补充附录，不是当前推荐主线。

### DeepShip UATR_KNN-C

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

### DeepShip ShuffleFAC

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

### ShipsEar UATR_KNN-C

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

### ShipsEar ShuffleFAC

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

### 暂停：DeepShip FA_UATR_KNN

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

### 暂停：ShipsEar FA_UATR_KNN

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
