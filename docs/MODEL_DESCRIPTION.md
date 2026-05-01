# Model Description

## MF_CONCAT

`MF_CONCAT` 是人工特征直接拼接版本。

- 输入：MFCC、Delta MFCC、STFT 统计特征。
- 处理方式：对时序特征做 mean / std pooling，得到固定长度向量。
- 分类器：MLP。
- 说明：该模型不使用 Log-Mel 作为深度主干输入，而是直接依赖人工统计特征。

## MF_BRANCH

`MF_BRANCH` 仍属于人工特征直接输入路线，但不是简单拼接原始特征。

- 输入：STFT、MFCC、Delta MFCC。
- 处理方式：
  - STFT 走 CNN 分支。
  - MFCC / Delta 走 CNN 分支。
  - 分支高层特征融合后分类。
- 说明：相比 `MF_CONCAT`，它增加了分支特征抽取，但实验上仍没有超过 Log-Mel 主输入路线。

## UATR_KNN

`UATR_KNN` 使用 Log-Mel 作为输入，关注 patch 之间的全局关系与局部相似关系。

流程：

```text
Log-Mel -> Patchify -> Patch tokens -> Transformer -> KNN graph -> MRGraphConv -> Pooling -> Classifier
```

A/B/C 变体：

- `A = Patch + Transformer`
- `B = Patch + KNN-GNN`
- `C = Patch + Transformer + KNN-GNN`

KNN 构图逻辑：

- 每个 Log-Mel patch token 是一个节点。
- 使用 `torch.cdist` 计算同一条样本内部 token 之间的欧氏距离。
- 每个节点选择最近 `K` 个邻居。
- 不跨样本构图，batch 内不同样本之间没有边。

MRGraphConv：

- 对邻居计算 `x_j - x_i`。
- 在邻居维做 max pooling，得到最大相对特征。
- 拼接自身特征和最大相对特征。
- 经过线性层、残差连接和 FFN 更新节点。

## ShuffleFAC

`ShuffleFAC` 是基于 Log-Mel 的轻量 CNN，不是 MFCC / MIPE 人工特征输入。

核心结构：

- FA block：通过频率位置编码和通道门控，使网络显式感知不同频带位置的信息差异。它不是只做普通卷积，而是把频率 bin 的位置信息注入到特征图中，让网络能区分不同频带上的模式。
- FASC：由 point-wise group convolution、depthwise convolution、channel shuffle、point-wise group convolution 组成，用轻量结构降低参数量与计算量。

当前工程接入的是 ShuffleFAC γ=16 配置：

```text
filters = [16, 32, 64, 128, 128, 128, 128]
```

输入输出：

- 输入：`[B, 1, F, T]` 的 Log-Mel spectrogram。
- 内部：转换为 `[B, C, T, F]` 以沿频率方向执行 FA/FASC 操作。
- 输出：全局池化后接线性分类器。

## UATR_KNN_REG

`UATR_KNN_REG` 的主输入仍然是 Log-Mel，主干为 `UATR_KNN-C`。

- 主任务：分类，使用 cross entropy。
- 辅助目标：MFCC / Delta / STFT 统计特征。
- Loss:

```text
loss = CE + lambda * auxiliary regression loss
```

当前实验未提升，说明人工特征作为辅助回归约束没有带来稳定收益。

## FA_UATR_KNN

`FA_UATR_KNN` 是后续计划模型，目标是融合 ShuffleFAC 的 FA/FASC 前端和 UATR_KNN 的 patch 关系建模能力。

计划结构：

```text
Log-Mel
-> FA/FASC Stem
-> Patch tokens
-> Transformer
-> KNN-MRGraphConv
-> Gated Residual Fusion
-> Pooling
-> Classifier
```

判断重点：

- 如果它超过 UATR_KNN-C，说明 FA block 能增强 UATR_KNN 前端表征。
- 如果它超过 ShuffleFAC，说明 FA block 与 KNN-GNN 有互补价值。
- 如果低于二者，则停止该融合路线。
