# Underwater Acoustic Target Recognition

本工程面向水声目标识别与船舶辐射噪声分类，主要在 DeepShip 和 ShipsEar 两个数据集上评估不同轻量模型在 **strict recording-level split** 下的泛化能力。当前实验重点不是 frame 级随机切片准确率，而是避免同一条录音的不同切片同时进入 train / val / test 后造成的 recording overlap。

## 数据集

- **DeepShip**：4 类，`Cargo / Passengership / Tanker / Tug`。
- **ShipsEar**：当前工程按 5 类处理，通常对应 `A / B / C / D / E` 或 `ClassA / ... / ClassE`。
- `--data_selection 0`：使用 DeepShip。
- `--data_selection 1`：使用 ShipsEar。

## 划分协议

工程支持两类 split：

- `frame_level`：直接按切片随机划分。该方式可能让同一原始录音的不同切片出现在 train / val / test 中，导致 recording overlap，结果通常偏乐观。
- `recording_level`：先按录音划分，再展开为切片。正式实验使用该协议。

正式结果必须检查：

```text
train-val recording overlap = 0
train-test recording overlap = 0
val-test recording overlap = 0
```

`Datasets/ShipsEar_dataloader.py` 会在 setup 后输出 split audit，并在 recording-level 模式下检测到 overlap 时直接停止训练。

## 当前主要模型

- `UATR_KNN`：Log-Mel 输入，Patch tokens + Transformer + KNN-MRGraphConv。
- `ShuffleFAC`：Log-Mel 输入，基于 FA block / FASC 的轻量 CNN。
- `MF_CONCAT`：MFCC / Delta MFCC / STFT 统计特征直接拼接。
- `MF_BRANCH`：人工特征多分支 CNN 融合。
- `UATR_KNN_REG`：Log-Mel 主干 + 人工特征辅助回归约束。
- `FA_UATR_KNN`：后续计划重点，尝试将 ShuffleFAC 的 FA/FASC 前端与 UATR_KNN 的 Transformer + KNN-GNN 结合。

## 当前核心结论

- 人工特征直接输入路线整体效果较弱。
- MIPE 加入后没有带来稳定提升。
- `UATR_KNN-C` 的 Transformer 与 KNN-GNN 组合有一定 patch 关系建模价值。
- `external/ShuffleFAC` 官方风格实现与工程内集成版 `src/models/shufflefac.py` 需要分开记录；当前复核显示 external ShuffleFAC 在 DeepShip `3s / 7:1:2` 下非常稳定。
- 后续应优先探索 `FA block + UATR_KNN` 的融合模型，而不是继续扩大 MFCC / MIPE / REG 路线。

## External ShuffleFAC 复核结果

以下结果均来自 `external/ShuffleFAC/run_deepship.py`，使用 `strict recording-level split`，
`segment_length=3`，`train/val/test=0.7/0.1/0.2`，`split_seed=42`，
训练种子为 `42/43/44`，特征参数为 `n_fft/win_length/hop_length/n_mels = 4096/4096/2048/128`。
所有 split audit 均为 recording overlap = 0。

### DeepShip

结果目录：`results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2`

| Seed | ACC | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: |
| 42 | 0.6829 | 0.6812 | 0.6828 |
| 43 | 0.6870 | 0.6868 | 0.6867 |
| 44 | 0.6831 | 0.6825 | 0.6845 |

- Mean ACC = 0.6843
- Mean Macro-F1 = 0.6835
- Sample Std Macro-F1 = 0.0029
- Mean Weighted-F1 = 0.6847
- Params = 39,031
- MACs = 2.585M

### ShipsEar

结果目录：`results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2`

| Seed | ACC | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: |
| 42 | 0.7073 | 0.5816 | 0.6892 |
| 43 | 0.7952 | 0.7059 | 0.7896 |
| 44 | 0.7412 | 0.6314 | 0.7295 |

- Mean ACC = 0.7479
- Mean Macro-F1 = 0.6396
- Sample Std Macro-F1 = 0.0626
- Mean Weighted-F1 = 0.7361
- Params = 39,160
- MACs = 2.585M

## 文档入口

- [实验结果汇总](docs/EXPERIMENT_SUMMARY.md)
- [模型说明](docs/MODEL_DESCRIPTION.md)
- [下一步计划](docs/NEXT_STEPS.md)
- [服务器运行命令](docs/RUN_COMMANDS.md)
- [MIPE+MFCC 单独运行说明](README_RUN_MIPE_MFCC.md)

## 快速运行示例

ShipsEar + UATR_KNN-C：

```bash
python demo_light.py \
  --model UATR_KNN \
  --uatr_variant C \
  --data_selection 1 \
  --split_protocol recording_level \
  --segment_length 5 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2
```

完整服务器命令见 [docs/RUN_COMMANDS.md](docs/RUN_COMMANDS.md)。
