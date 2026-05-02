# Next Steps

当前阶段的主线已经从 `FA_UATR_KNN / FA_UATR_KNN_V2` 转为：

```text
external ShuffleFAC native 3s clips
-> pretrained clip-level frequency-aware encoder
-> recording-level aggregation
```

核心目标不是继续堆时频 patch Transformer/GNN，而是证明同一 recording 内多段 clips 的聚合能提升 recording-level 识别稳定性。

## 当前主结果

### DeepShip

DeepShip 主结果保留：

```text
ShuffleFAC native 3s/7:1:2 + ordinary recording-level mean-logit voting
```

| Method | Mean Recording Macro-F1 | Sample Std | 结论 |
| --- | ---: | ---: | --- |
| Ordinary voting | 0.7730 | 0.0091 | 当前主结果 |
| Frozen AttentionHead | 0.7711 | 0.0537 | 接近 voting，但方差更大 |
| Frozen GraphHead | 0.7707 | 0.0290 | GNN 消融 |
| Graph-aware AttentionHead | 0.7725 | 0.0178 | 接近主结果，可作为 GNN-guided 消融 |

### ShipsEar

ShipsEar 主结果保留：

```text
pretrained external ShuffleFAC encoder
-> frozen encoder
-> deterministic multi-sample AttentionHead
```

| Method | Mean Recording Macro-F1 | Sample Std | 结论 |
| --- | ---: | ---: | --- |
| Ordinary voting | 0.6255 | 0.1003 | 不稳定 baseline |
| Frozen AttentionHead | 0.6876 | 0.0841 | 当前主结果 |
| Frozen GraphHead | 0.6356 | 0.0780 | GNN 消融 |
| Graph-aware AttentionHead | 0.6765 | 0.1029 | 高于 voting，低于 AttentionHead |

## 下一步优先级

1. 固化论文/报告表格：DeepShip 用 ordinary recording voting；ShipsEar 用 frozen AttentionHead。
2. 在方法部分明确区分 clip-level ShuffleFAC encoder 和 recording-level aggregation head。
3. 报告 GraphHead / Graph-aware AttentionHead 作为 GNN-guided recording aggregation 消融，而不是当前主结果。
4. 检查最终表格是否都注明：`3s / 7:1:2 / strict recording-level split / eval_samples=5`。
5. 保留 split audit：正式结果必须满足 train/val/test recording overlap = 0。

## 暂停路线

当前不要继续投入：

- `FA_UATR_KNN_V2` 结构优化。
- from-scratch `ShuffleFAC_CLIPGRAPH` 200 epoch / 3-seed。
- `UATR_KNN_REG`、MFCC、MIPE、多人工特征输入。
- Transformer/GNN 重新插回时频 patch 的复杂融合。
- SpecAugment、contrastive loss、weighted CE 等额外训练技巧。

## 可选后续

如果老师希望继续保留 GNN 故事线，优先做轻量、可解释的消融：

- `graph_aware_attention` 的 edge mode 消融：`temporal` / `similarity` / `temporal_similarity`。
- Graph-aware AttentionHead 与普通 AttentionHead 的 attention entropy 对比。
- DeepShip 上 Graph-aware AttentionHead 是否能在不同 `eval_samples` 下保持接近 ordinary voting。

这些都应作为后续分析，不应替代当前主结果。
