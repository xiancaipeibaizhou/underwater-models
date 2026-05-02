# FA_UATR_KNN_V2 Negative Ablation Note

## Setup

All ShipsEar comparisons below use the same strict recording-level split:

- `split_seed = 42`
- `train/val/test = 0.7/0.1/0.2`
- `segment_length = 5`
- `sample_rate/window/hop/mels = 16000/2048/512/128`
- Training seeds are `model_seed + run_index = 42/43/44`

The split is fixed across models and runs. Therefore, the reported variation mainly reflects training-seed, initialization, dataloader-order, and checkpoint-selection effects rather than split randomness.

## ShipsEar Fixed-Split Results

| Model | Runs Test Macro-F1 | Mean | Std | Params | MACs |
| --- | --- | ---: | ---: | ---: | ---: |
| ShuffleFAC | 0.5458 / 0.5303 / 0.2730 | 0.4497 | 0.1532 | 110,696 | 51.330M |
| UATR_KNN-C | 0.5963 / 0.6412 / 0.6228 | 0.6201 | 0.0226 | 362,837 | 39.431M |
| FA_UATR_KNN_V2 post_trans gate=-4 gated | 0.5526 / 0.4657 / 0.5653 | 0.5279 | 0.0542 | 967,917 | 73.749M |

Std is computed as sample standard deviation over three runs.

The earlier single ShuffleFAC run reached 0.6916 Test Macro-F1. However, the repeated fixed-split runs show that this value should be treated as a historical best single-run result rather than a stable multi-seed estimate. Under the fixed split, UATR_KNN-C is the strongest and most stable model among the compared ShipsEar baselines.

## Interpretation

FA_UATR_KNN_V2 does not provide a robust gain over UATR_KNN-C on ShipsEar. Its best seed is still below the UATR_KNN-C mean, while using substantially more parameters and MACs.

The diagnostic V2 runs also show that the Transformer-only path is weak. The gated Graph branch can partially improve the representation, but it does not recover enough performance to justify further structure tuning.

Therefore, FA_UATR_KNN_V2 should be reported as a negative ablation. Adding the FASCStem plus gated graph fusion increases model complexity but does not improve fixed-split performance. Further optimization of this route should stop. Future work should focus on stronger and stabler baselines, especially UATR_KNN-C on ShipsEar and the best-performing lightweight CNN baseline on DeepShip.