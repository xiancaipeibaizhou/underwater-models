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

结论：ShuffleFAC 在 DeepShip 上略高于 UATR_KNN-C，并且复杂度更低。

## 6. ShipsEar 对比结果

### UATR_KNN-C

- ACC = 0.6757
- Macro-F1 = 0.5831
- Weighted-F1 = 0.6603
- Params = 362,837
- MACs = 39.431M
- Latency = 1.461 ms/batch1

### ShuffleFAC gamma=16

- ACC = 0.6982
- Macro-F1 = 0.6916
- Weighted-F1 = 0.7200
- Params = 110,696
- MACs = 51.330M
- Latency = 3.133 ms/batch1

结论：ShuffleFAC 在 ShipsEar 上也明显优于 UATR_KNN-C，尤其 Macro-F1 提升明显，说明 FA/FASC block 具有跨数据集参考价值。
