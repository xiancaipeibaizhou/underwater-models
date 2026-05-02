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

## 13. UATR_KNN-C native 3s/7:1:2 seed42 sanity

Setting: `3s / 7:1:2 / 4096-2048 / 128 Mel / 16 kHz / strict recording-level split`.
This is a seed42 sanity run with `patience=40` to verify the native-protocol command,
recording-level metrics, and split audit before launching any 3-seed run.

| Dataset | Seed | Segment ACC | Segment Macro-F1 | Segment Weighted-F1 | Recording ACC | Recording Macro-F1 | Recording Weighted-F1 | Params | MACs | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepShip | 42 | 0.6235 | 0.6213 | 0.6220 | 0.7131 | 0.6698 | 0.7071 | 362,740 | 6.507M | 1.322 ms |
| ShipsEar | 42 | 0.7015 | 0.6583 | 0.7024 | 0.6667 | 0.6429 | 0.6455 | 362,837 | 6.507M | 1.345 ms |

Split audit: both runs have `train-val`, `train-test`, and `val-test` recording overlap = 0.

结论：UATR_KNN-C native seed42 的命令和 recording-level metrics 均已跑通。DeepShip 上
seed42 recording Macro-F1 = 0.6698，低于 ShuffleFAC native + mean-logit voting 的
3-seed mean 0.7729；因此 DeepShip 当前仍应以 ShuffleFAC + recording-level voting
作为主结果。ShipsEar seed42 的 UATR_KNN-C recording Macro-F1 = 0.6429，略高于
ShuffleFAC voting seed42 但仍需 3-seed 才能判断稳定性。

## 14. ShuffleFAC_GRAPHHEAD sanity

Setting: DeepShip seed42, pretrained external ShuffleFAC `best.pt`, frozen encoder,
`clips_per_recording=8`, `3s / 7:1:2 / 4096-2048 / 128 Mel / 16 kHz`,
`batch_size=8`, `lr=1e-3`, `weight_decay=1e-4`, `num_epochs=50`, `patience=20`.

Pretrained checkpoint:
`results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_42/best.pt`

Ordinary ShuffleFAC mean-logit voting baseline for this seed:
Recording Macro-F1 = 0.7628.

| Head | Best Val Macro-F1 | Test Recording ACC | Test Recording Macro-F1 | Test Recording Weighted-F1 | attn_entropy | graph_delta_norm | graph_res_scale | Trainable Params | Frozen Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention | 0.8518 | 0.7541 | 0.7619 | 0.7536 | 1.8525 | - | - | 9,093 | 39,031 |
| Graph | 0.8664 | 0.7787 | 0.7844 | 0.7795 | 1.9976 | 2.3707 | 0.1956 | 58,758 | 39,031 |

结论：Attention head 几乎追平 ordinary voting，但未超过 seed42 baseline。Graph head
超过 attention head，并超过 ordinary voting baseline，说明在 frozen ShuffleFAC embedding
上做 lightweight graph relation aggregation 有初步价值。下一步应优先在 ShipsEar seed42/44
验证该方向是否能缓解 voting instability，而不是回到 from-scratch ClipGraph 或继续 V2。

## 15. ShuffleFAC_GRAPHHEAD deterministic multi-sample evaluation

Implementation note: validation/test sampling is deterministic. The default evaluation uses evenly
spaced `S=8` clips per recording. The multi-sample setting uses `eval_samples=5`, where each recording
is evaluated with five deterministic temporal phases and logits are averaged before computing
recording-level metrics. Strategy selection is still based on validation metrics, not test metrics.

### DeepShip 3-seed, frozen encoder, `eval_samples=5`

| Seed | Ordinary Voting Macro-F1 | Attention Best Val | Attention Test Macro-F1 | Attention ACC | Graph Best Val | Graph Test Macro-F1 | Graph ACC | Graph Delta Norm | Graph Res Scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7628 | 0.8650 | 0.8328 | 0.8279 | 0.8547 | 0.8013 | 0.7951 | 3.3087 | 0.2640 |
| 43 | 0.7804 | 0.8584 | 0.7449 | 0.7541 | 0.8756 | 0.7672 | 0.7787 | 3.0520 | 0.2453 |
| 44 | 0.7757 | 0.8822 | 0.7355 | 0.7459 | 0.8564 | 0.7437 | 0.7459 | 2.7454 | 0.2200 |

| Method | Mean Test Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.7730 | 0.0091 |
| AttentionHead | 0.7711 | 0.0537 |
| GraphHead | 0.7707 | 0.0290 |

结论：DeepShip 上 GraphHead 没有稳定超过 ordinary voting。GraphHead 相比 AttentionHead
更稳，但均值仍略低于 ordinary voting；因此 DeepShip 主结果仍保留 ShuffleFAC native
mean-logit voting。

### DeepShip Graph-aware AttentionHead 3-seed

Graph-aware AttentionHead uses graph context only to compute clip attention weights:
`attn_score = MLP(concat(z, graph_context))`, while the final recording embedding
still pools the original frozen ShuffleFAC clip embeddings `z`.

| Seed | Ordinary Voting Macro-F1 | Graph-aware Best Val | Graph-aware Test Macro-F1 | Graph-aware ACC | attn_entropy | graph_delta_norm | graph_res_scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7628 | 0.8410 | 0.7896 | 0.7951 | 1.5346 | 1.6210 | 0.1549 |
| 43 | 0.7804 | 0.8624 | 0.7541 | 0.7705 | 1.4627 | 1.9266 | 0.2030 |
| 44 | 0.7757 | 0.8720 | 0.7737 | 0.7787 | 1.2230 | 1.7905 | 0.2096 |

| Method | Mean Test Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.7730 | 0.0091 |
| AttentionHead | 0.7711 | 0.0537 |
| GraphHead | 0.7707 | 0.0290 |
| Graph-aware AttentionHead | 0.7725 | 0.0178 |

结论：DeepShip Graph-aware AttentionHead 几乎追平 ordinary voting，但没有超过，
且方差没有低于 ordinary voting。因此 DeepShip 主结果仍保留 ShuffleFAC native
mean-logit voting；Graph-aware AttentionHead 可作为接近主结果的 GNN 消融。

### ShipsEar seed42/44, frozen encoder, `eval_samples=5`

| Seed | Ordinary Voting Macro-F1 | Attention Best Val | Attention Test Macro-F1 | Attention ACC | Graph Best Val | Graph Test Macro-F1 | Graph ACC | Graph Delta Norm | Graph Res Scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.5505 | 0.8200 | 0.5933 | 0.6842 | 0.8200 | 0.5588 | 0.6316 | 2.5870 | 0.2094 |
| 44 | 0.5867 | 0.9048 | 0.7148 | 0.7368 | 0.9048 | 0.7148 | 0.7368 | 1.3232 | 0.1158 |

| Method | Mean Test Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.5686 | 0.0256 |
| AttentionHead | 0.6540 | 0.0859 |
| GraphHead | 0.6368 | 0.1103 |

结论：ShipsEar seed42/44 上 learned aggregation 明显高于 ordinary voting，尤其
AttentionHead 更强。GraphHead 没有超过 AttentionHead；当前证据更支持 frozen
ShuffleFAC encoder + deterministic multi-sample AttentionHead，而不是更复杂的 graph head。

### ShipsEar AttentionHead 3-seed completion

| Seed | Ordinary Voting Macro-F1 | Attention Best Val | Attention Test Macro-F1 | Attention ACC | attn_entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.5505 | 0.8200 | 0.5933 | 0.6842 | 1.9685 |
| 43 | 0.7394 | 0.8933 | 0.7548 | 0.7895 | 2.0409 |
| 44 | 0.5867 | 0.9048 | 0.7148 | 0.7368 | 2.0470 |

| Method | Mean Test Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.6255 | 0.1003 |
| AttentionHead | 0.6876 | 0.0841 |

结论：ShipsEar 上 frozen ShuffleFAC encoder + deterministic multi-sample AttentionHead
相对 ordinary voting 同时提升均值并降低方差，是当前最值得保留的 recording-level
aggregation 方向。

### ShipsEar GraphHead 3-seed completion

| Seed | Ordinary Voting Macro-F1 | Graph Best Val | Graph Test Macro-F1 | Graph ACC | attn_entropy | graph_delta_norm | graph_res_scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.5505 | 0.8200 | 0.5588 | 0.6316 | 1.8772 | 2.5870 | 0.2094 |
| 43 | 0.7394 | 0.9048 | 0.6333 | 0.6842 | 2.0677 | 1.3241 | 0.1161 |
| 44 | 0.5867 | 0.9048 | 0.7148 | 0.7368 | 2.0595 | 1.3232 | 0.1158 |

| Method | Mean Test Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.6255 | 0.1003 |
| AttentionHead | 0.6876 | 0.0841 |
| GraphHead | 0.6356 | 0.0780 |

结论：ShipsEar GraphHead 3-seed 均值略高于 ordinary voting，方差更低，但明显低于
AttentionHead。当前不支持引入 graph relation 作为主线；更合理的主线是 frozen
ShuffleFAC encoder + deterministic multi-sample AttentionHead。

### ShipsEar Graph-aware AttentionHead 3-seed

Graph-aware AttentionHead uses graph context only to compute clip attention weights:
`attn_score = MLP(concat(z, graph_context))`, while the final recording embedding
still pools the original frozen ShuffleFAC clip embeddings `z`.

| Seed | Ordinary Voting Macro-F1 | Graph-aware Best Val | Graph-aware Test Macro-F1 | Graph-aware ACC | attn_entropy | graph_delta_norm | graph_res_scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.5505 | 0.8933 | 0.5600 | 0.6316 | 1.5729 | 1.9621 | 0.1778 |
| 43 | 0.7394 | 0.8933 | 0.7548 | 0.7895 | 1.8548 | 1.3261 | 0.1193 |
| 44 | 0.5867 | 0.9048 | 0.7148 | 0.7368 | 1.8737 | 1.3384 | 0.1204 |

| Method | Mean Test Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.6255 | 0.1003 |
| AttentionHead | 0.6876 | 0.0841 |
| GraphHead | 0.6356 | 0.0780 |
| Graph-aware AttentionHead | 0.6765 | 0.1029 |

结论：Graph-aware AttentionHead 高于 ordinary voting，但低于 AttentionHead，且方差没有降低。
因此 GNN 当前只能作为消融保留；ShipsEar 主线仍是 frozen ShuffleFAC encoder +
deterministic multi-sample AttentionHead。
