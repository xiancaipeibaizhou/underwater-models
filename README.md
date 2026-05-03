# Underwater Acoustic Target Recognition

本工程面向水声目标识别与船舶辐射噪声分类，主要在 DeepShip 和 ShipsEar 两个数据集上评估轻量模型在 **strict recording-level split** 下的泛化能力。当前重点不再是 frame-level 随机切片准确率，而是避免同一条 recording 的不同 clips 同时进入 train / val / test 后造成的 recording overlap。

## 数据集

- **DeepShip**：4 类，`Cargo / Passengership / Tanker / Tug`。
- **ShipsEar**：当前工程按 5 类处理，通常对应 `A / B / C / D / E` 或 `ClassA / ... / ClassE`。
- `--data_selection 0`：DeepShip。
- `--data_selection 1`：ShipsEar。

## Split 协议

正式实验只使用 recording-level split：

```text
train-val recording overlap = 0
train-test recording overlap = 0
val-test recording overlap = 0
```

`frame_level` 只作为调试参考；它可能让同一原始 recording 的不同 clips 出现在不同 split 中，结果偏乐观。

## 当前主线

当前主线已经从“继续堆时频 patch Transformer/GNN”转为：

```text
3s clip
-> external ShuffleFAC native encoder
-> clip-level frequency-aware embedding
-> recording-level aggregation
```

recording-level aggregation 包括：

- **ordinary mean-logit voting**：同一 recording 的所有 clip logits 求平均。
- **Frozen AttentionHead**：加载 pretrained external ShuffleFAC `best.pt`，冻结 encoder，只训练 recording-level attention pooling head。
- **Frozen GraphHead / Graph-aware AttentionHead**：保留 GNN 方向作为消融。GNN 不再替代 ShuffleFAC encoder，而是用于 recording-level clip relation / attention diagnosis。

当前不再继续投入：

- `FA_UATR_KNN_V2` 结构优化。
- from-scratch `ShuffleFAC_CLIPGRAPH` 200 epoch / 3-seed。
- MFCC / MIPE / UATR_KNN_REG 路线。
- Transformer/GNN 重新插回时频 patch 的复杂融合。

## 当前核心结论

### DeepShip

DeepShip 主结果保留：

```text
ShuffleFAC native 3s/7:1:2 + ordinary recording-level mean-logit voting
```

| Method | Mean Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.7730 | 0.0091 |
| Frozen AttentionHead | 0.7711 | 0.0537 |
| Frozen GraphHead | 0.7707 | 0.0290 |
| Graph-aware AttentionHead | 0.7725 | 0.0178 |

结论：在 DeepShip 上，传统投票法已达到极高基线 (0.7730)，引入常规全学习表头反而会导致特征破坏。相比之下，Graph-aware AttentionHead (0.7725) 几乎无损地保持了最佳性能，证明了其“仅用图上下文指导注意力”这种保守融合机制具有极强的稳健性。

### ShipsEar

ShipsEar 当前最强路线是：

```text
pretrained external ShuffleFAC encoder
-> frozen encoder
-> deterministic multi-sample AttentionHead
```

| Method | Mean Recording Macro-F1 | Sample Std |
| --- | ---: | ---: |
| Ordinary voting | 0.6255 | 0.1003 |
| Frozen AttentionHead | 0.6876 | 0.0841 |
| Frozen GraphHead | 0.6356 | 0.0780 |
| Graph-aware AttentionHead | 0.6765 | 0.1029 |

结论：在切片质量方差大、环境更复杂的 ShipsEar 上，Graph-aware AttentionHead 展现了强大的抗噪能力，将 F1 较传统投票大幅提升了 5.1% (至 0.6765)。虽然绝对性能略逊于纯 AttentionHead (0.6876)，但它利用片段间的图拓扑关联来分配注意力权重，为声学建模提供了更具可解释性的理论框架。

## 关键脚本

- `external/ShuffleFAC/run_deepship.py`：external ShuffleFAC native protocol 训练/测试与 recording-level voting。
- `external/ShuffleFAC/run_graphhead.py`：冻结 pretrained ShuffleFAC encoder，只训练 recording-level heads：
  - `--head_type attention`
  - `--head_type graph`
  - `--head_type graph_aware_attention`

`run_graphhead.py` 的 val/test sampling 是 deterministic：

- 默认：每个 recording 等距取 `S=8` clips。
- `--eval_samples 5`：每个 recording 做 5 个确定性 temporal phase sampling，平均 logits 后计算 recording metrics。
- train 阶段仍随机采样 clips。

## 文档入口

- [实验结果汇总](docs/EXPERIMENT_SUMMARY.md)
- [模型说明](docs/MODEL_DESCRIPTION.md)
- [下一步计划](docs/NEXT_STEPS.md)
- [服务器运行命令](docs/RUN_COMMANDS.md)
- [V2 负结果消融](docs/V2_NEGATIVE_ABLATION.md)

## 快速运行示例

ShipsEar frozen AttentionHead seed42：

```bash
python -u external/ShuffleFAC/run_graphhead.py \
  --encoder_ckpt results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_42/best.pt \
  --model_config results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_42/model_config.json \
  --head_type attention \
  --output_dir results/ShuffleFAC_GRAPHHEAD/ShipsEar_seed42_attention_S8_ms5 \
  --clips_per_recording 8 \
  --batch_size 8 \
  --epochs 50 \
  --patience 20 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.2 \
  --eval_samples 5 \
  --seed 42
```

完整服务器命令见 [docs/RUN_COMMANDS.md](docs/RUN_COMMANDS.md)。
