# Next Steps

## 停止推进的路线

当前阶段先停止以下方向：

- MFCC / Delta / STFT / MIPE 直接输入。
- MIPE scale=10。
- 继续扩大 `UATR_KNN_REG` 参数规模。
- `MF_CROSSATTN` / `MF_KNN` 多特征输入路线。

原因：已有实验显示这些路线没有超过 Log-Mel 主输入路线，继续堆复杂度的优先级较低。

## 当前主线

当前主线是参考 ShuffleFAC 的 FA/FASC block，将其前端频率自适应建模能力与 UATR_KNN 的 Transformer + KNN-GNN 关系建模能力结合。

目标模型：

```text
FA_UATR_KNN
```

设计结构：

```text
Log-Mel
-> FA/FASC Stem
-> Patch tokens
-> Transformer
-> KNN-GNN
-> Gated Residual Fusion
-> Classifier
```

## 判断标准

- `FA_UATR_KNN > UATR_KNN-C`：说明 FA block 能增强 UATR_KNN 前端。
- `FA_UATR_KNN > ShuffleFAC`：说明 FA block 与 KNN-GNN 有互补价值。
- `FA_UATR_KNN` 介于二者之间：说明 FA block 有帮助，但图关系建模或融合方式还需要优化。
- `FA_UATR_KNN` 低于二者：停止该融合路线。

## 实验注意事项

- 继续使用 strict recording-level split。
- 每次正式结果都必须确认 recording overlap 为 0。
- 不再把 frame-level 随机切分作为公平最终结果。
- 不引入预训练、知识蒸馏或额外数据。
