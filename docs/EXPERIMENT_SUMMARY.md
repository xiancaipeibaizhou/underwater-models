# Experiment Summary

本文档只保留当前阶段的重要实验结果，不记录临时 debug 和 smoke test。

## 1. DeepShip 人工特征路线

### MF_CONCAT no MIPE

- ACC = 0.5774
- Macro-F1 = 0.5769
- Weighted-F1 = 0.5804

### MF_BRANCH no MIPE

- ACC = 0.5764
- Macro-F1 = 0.5734
- Weighted-F1 = 0.5724

### MF_CONCAT + MIPE scale=5

- ACC 约 0.5630

结论：人工特征作为直接输入效果较弱，MIPE 未提升。

## 2. DeepShip UATR_KNN A/B/C 消融

### A = Patch + Transformer

- ACC = 0.6167
- Macro-F1 = 0.6146
- Weighted-F1 = 0.6131

### B = Patch + KNN-GNN

- ACC = 0.6236
- Macro-F1 = 0.6219
- Weighted-F1 = 0.6199

### C = Patch + Transformer + KNN-GNN

- ACC = 0.6357
- Macro-F1 = 0.6342
- Weighted-F1 = 0.6336

说明：A/B/C 呈现 `C > B > A`，说明 Transformer 全局建模与 KNN-GNN 局部关系建模存在互补性。

## 3. DeepShip UATR_KNN 200 epoch 对齐结果

### UATR_KNN-C

- ACC = 0.6214
- Macro-F1 = 0.6187
- Weighted-F1 = 0.6172
- Params = 362,740
- MACs = 39.431M
- Latency = 1.431 ms/batch1

## 4. DeepShip UATR_KNN_REG

### UATR_KNN_REG aux=0.05

- ACC = 0.6054
- Macro-F1 = 0.6018
- Weighted-F1 = 0.5997
- Params = 421,700
- MACs = 39.431M

结论：人工特征作为辅助回归约束也没有提升，反而低于 UATR_KNN-C。

## 5. DeepShip ShuffleFAC 对齐结果

### ShuffleFAC gamma=16

Setting: `5s / 6:2:2 / 2048-512 / strict recording-level`

- ACC = 0.6419
- Macro-F1 = 0.6426
- Weighted-F1 = 0.6435
- Params = 39,031
- MACs = 16.792M

说明：这是早期单 seed 对齐结果。后续观察到 DeepShip 上存在 split-protocol sensitivity：
ShuffleFAC 在 `7:1:2` 划分下曾更好，而 UATR_KNN-C 在 `6:2:2` 下更好。因此不同
train/val/test 比例下的结果只能视为“模型 + split 协议”的组合表现，不能直接用于模型稳定性比较。

## 6. DeepShip fixed-split 3-seed 对比

Setting: `5s / 6:2:2 / split_seed=42 / model_seed=42+run_index / strict recording-level`

| Model | Run 0 Macro-F1 | Run 1 Macro-F1 | Run 2 Macro-F1 | Mean Macro-F1 | Std | Mean ACC | Mean Weighted-F1 | Params | MACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShuffleFAC gamma=16 | 0.5768 | 0.5950 | 0.5735 | 0.5818 | 0.0095 | 0.5833 | 0.5811 | 110,567 | 51.330M |
| UATR_KNN-C | 0.5941 | 0.6296 | 0.5593 | 0.5943 | 0.0287 | 0.5965 | 0.5913 | 362,740 | 39.431M |

结论：在 DeepShip 的 `6:2:2` fixed split 下，UATR_KNN-C 平均 Macro-F1 略高，
但方差也明显更大；ShuffleFAC 更稳定，但稳定在较低水平。若讨论“各模型最佳 split 协议”，
需要单独跑 ShuffleFAC 的 `7:1:2` 3-seed 表，不能和 UATR 的 `6:2:2` 表直接混合比较。

## 7. ShipsEar 对比结果

### 7.1 Historical single-run results

Setting: `5s / 7:1:2 / 2048-512 / strict recording-level`

#### UATR_KNN-C

- ACC = 0.6757
- Macro-F1 = 0.5831
- Weighted-F1 = 0.6603
- Params = 362,837
- MACs = 39.431M
- Latency = 1.461 ms/batch1

#### ShuffleFAC gamma=16

- ACC = 0.6982
- Macro-F1 = 0.6916
- Weighted-F1 = 0.7200
- Params = 110,696
- MACs = 51.330M
- Latency = 3.133 ms/batch1

说明：ShuffleFAC 的 `Macro-F1 = 0.6916` 是早期 historical best single-run。后续 fixed-split repeated training 表明，该结果不能作为稳定 multi-seed 平均性能直接使用。

### 7.2 Fixed-split 3-seed comparison

Setting: `split_seed=42 / train-val-test=0.7:0.1:0.2 / model_seed=42+run_index / strict recording-level`

| Model | Run 0 Macro-F1 | Run 1 Macro-F1 | Run 2 Macro-F1 | Mean Macro-F1 | Std | Params | MACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShuffleFAC gamma=16 | 0.5458 | 0.5303 | 0.2730 | 0.4497 | 0.1532 | 110,696 | 51.330M |
| UATR_KNN-C | 0.5963 | 0.6412 | 0.6228 | 0.6201 | 0.0226 | 362,837 | 39.431M |
| FA_UATR_KNN_V2 post_trans gate=-4 gated | 0.5526 | 0.4657 | 0.5653 | 0.5279 | 0.0542 | 967,917 | 73.749M |

说明：Std 使用三次运行的 sample standard deviation。若使用 population standard deviation，需要在表注中明确说明。

结论：在 ShipsEar fixed-split 3-seed 对比中，UATR_KNN-C 平均 Macro-F1 最高，且方差最小，是当前 ShipsEar 上最稳定的模型。ShuffleFAC 的早期单次高点较高，但在固定 split 重复训练下波动明显。FA_UATR_KNN_V2 的 gated graph 分支能部分提升 trans_only 主干，但没有稳定超过 UATR_KNN-C，且参数量和计算量更高，因此不再继续优化。

## 8. FA_UATR_KNN_V2 Negative Ablation

### Setup

All ShipsEar comparisons use the same fixed recording-level split:

- `split_seed = 42`
- `train/val/test = 0.7/0.1/0.2`
- `segment_length = 5`
- `sample_rate/window/hop/mels = 16000/2048/512/128`
- Training seeds are `model_seed + run_index = 42/43/44`

### Results

| Model | Runs Test Macro-F1 | Mean | Std | Params | MACs |
| --- | --- | ---: | ---: | ---: | ---: |
| UATR_KNN-C | 0.5963 / 0.6412 / 0.6228 | 0.6201 | 0.0226 | 362,837 | 39.431M |
| FA_UATR_KNN_V2 post_trans gate=-4 gated | 0.5526 / 0.4657 / 0.5653 | 0.5279 | 0.0542 | 967,917 | 73.749M |
| FA_UATR_KNN_V2 trans_only | 0.4810 | - | - | 967,917 | 73.749M |

### Interpretation

FA_UATR_KNN_V2 does not provide a robust gain over UATR_KNN-C on ShipsEar. The Transformer-only path is weak, while the gated graph branch can partially improve the representation. However, the improvement is insufficient to surpass UATR_KNN-C, and the model introduces significantly higher complexity.

Therefore, FA_UATR_KNN_V2 is retained as a negative ablation. Adding FASCStem plus gated graph fusion increases parameters and MACs but does not improve fixed-split performance. Further optimization of this route is stopped.

## 9. ShuffleFAC native 3s/7:1:2 + recording-level voting

Setting: `3s / 7:1:2 / 4096-2048 / 128 Mel / 16 kHz / strict recording-level split`.

注意：本节是 recording-level metrics。它们来自同一 recording 内多个 3s clips 的
mean-logit aggregation，不应与论文 clip-level metrics 混写。

### DeepShip

| Seed | Segment Macro-F1 | Recording Macro-F1 | Recording ACC |
| ---: | ---: | ---: | ---: |
| 42 | 0.6812 | 0.7628 | 0.7705 |
| 43 | 0.6868 | 0.7804 | 0.7869 |
| 44 | 0.6825 | 0.7757 | 0.7705 |

- Mean Recording Macro-F1 = 0.7729
- Sample Std = 0.0091

结论：DeepShip 上 recording-level mean-logit voting 显著提升，并已超过 0.70。
因此 ShuffleFAC native + recording-level voting 是当前 DeepShip 的强 baseline。

### ShipsEar

| Seed | Segment Macro-F1 | Recording Macro-F1 | Recording ACC |
| ---: | ---: | ---: | ---: |
| 42 | 0.5816 | 0.5505 | 0.6316 |
| 43 | 0.7059 | 0.7394 | 0.7895 |
| 44 | 0.6314 | 0.5867 | 0.6842 |

- Mean Recording Macro-F1 = 0.6255
- Sample Std = 0.1003

结论：ShipsEar 上 mean-logit voting 不稳定，平均 Recording Macro-F1 低于
segment-level mean，因此需要进一步诊断 aggregation strategy。正式选择聚合策略时应基于
validation split，而不是 test split。

## 10. ShuffleFAC_CLIPGRAPH smoke result

### DeepShip ShuffleFAC_CLIPGRAPH S=4 20 epoch smoke

- Best Val Macro-F1 = 0.6223
- Test Macro-F1 = 0.4527
- Test Recording Macro-F1 = 0.4527
- attn_entropy best = 1.3248, final = 1.2852
- graph_delta_norm best = 2.2434, final = 2.3842
- graph_res_scale = 0.1849
- Params = 169,069
- MACs = 31.803M
- Latency = 4.426 ms/batch1

结论：Current from-scratch ShuffleFAC_CLIPGRAPH S=4 is not competitive with ordinary
ShuffleFAC + recording-level voting. Therefore, we do not run 200-epoch 3-seed
ClipGraph training at this stage.

## 11. Future option: ShuffleFAC-GraphHead

暂时不执行，仅作为后续可选方向保留：

- load pretrained ShuffleFAC `best.pt`
- freeze ShuffleFAC encoder
- train only graph aggregation head
- compare against ordinary recording-level voting

该方向需要等 aggregation strategy evaluation 和 UATR_KNN-C native baseline 完成后再决定。

## 12. Recording aggregation strategy evaluation

Setting: ShuffleFAC native `3s / 7:1:2 / 4096-2048 / 128 Mel / 16 kHz / strict recording-level split`.
All runs are test-only evaluation from existing ShuffleFAC `best.pt` checkpoints. Strategy selection should
be based on validation metrics, not test metrics.

### DeepShip aggregation summary

| Strategy | Val Macro-F1 Mean | Val Std | Test Macro-F1 Mean | Test Std |
| --- | ---: | ---: | ---: | ---: |
| mean_logits | 0.8375 | 0.0245 | 0.7729 | 0.0091 |
| mean_probs | 0.8472 | 0.0179 | 0.7693 | 0.0143 |
| majority_vote | 0.8389 | 0.0243 | 0.7686 | 0.0072 |
| topk_confident_logits_25 | 0.8377 | 0.0269 | 0.7588 | 0.0148 |
| topk_confident_logits_50 | 0.8463 | 0.0285 | 0.7620 | 0.0188 |
| topk_confident_logits_75 | 0.8416 | 0.0203 | 0.7693 | 0.0237 |
| entropy_filtered_logits | 0.8463 | 0.0285 | 0.7653 | 0.0203 |
| trimmed_mean_logits | 0.8416 | 0.0203 | 0.7729 | 0.0091 |

DeepShip 上 validation 最优为 `mean_probs`，但 test 上 `mean_logits` 与
`trimmed_mean_logits` 最高且更稳。由于不能用 test 选策略，当前仍保留
`mean_logits` 作为默认强 baseline，并把 `mean_probs` 作为后续验证候选。

### ShipsEar aggregation summary

| Strategy | Val Macro-F1 Mean | Val Std | Test Macro-F1 Mean | Test Std |
| --- | ---: | ---: | ---: | ---: |
| mean_logits | 0.8405 | 0.0563 | 0.6255 | 0.1003 |
| mean_probs | 0.7841 | 0.1293 | 0.6142 | 0.1140 |
| majority_vote | 0.7841 | 0.1293 | 0.6142 | 0.1140 |
| topk_confident_logits_25 | 0.7905 | 0.1193 | 0.6100 | 0.1194 |
| topk_confident_logits_50 | 0.8405 | 0.0563 | 0.6255 | 0.1003 |
| topk_confident_logits_75 | 0.7849 | 0.1280 | 0.6255 | 0.1003 |
| entropy_filtered_logits | 0.8405 | 0.0563 | 0.6255 | 0.1003 |
| trimmed_mean_logits | 0.8405 | 0.0563 | 0.6142 | 0.1140 |

ShipsEar 上没有发现比 `mean_logits` 更稳的 aggregation method。Validation 上
`mean_logits`、`topk_confident_logits_50`、`entropy_filtered_logits` 和
`trimmed_mean_logits` 并列，但 test 上没有带来稳定提升；ShipsEar 的主要问题仍是
seed/split sensitivity 和 recording 数量较少，而不是简单聚合函数选择。
