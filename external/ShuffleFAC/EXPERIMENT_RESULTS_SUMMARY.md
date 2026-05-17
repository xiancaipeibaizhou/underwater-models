# ShuffleFAC 实验结果总表

更新时间：2026-05-17

本文件汇总当前项目中 ShuffleFAC 相关实验的主要结果。除特别说明外，主指标均为 strict recording-level test split 上的 recording-level 指标，不混入 clip-level 或 segment-level 指标。多 seed 统计使用 seed 42/43/44。

## 表 1：多 Seed 汇总指标

| 序号 | 实验组 | 数据集 | 实验/模型 | 阶段 | Seeds | ACC mean +/- std | Macro-F1 mean +/- std | Weighted-F1 mean +/- std | Params | Trainable Params | MACs | 备注 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ShuffleFAC encoder | DeepShip | recording_eval | Stage-1 frozen encoder eval | 3 | 0.775956 +/- 0.007728 | 0.772940 +/- 0.007432 | 0.776559 +/- 0.007413 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 2 | ShuffleFAC encoder | DeepShip | aggregation_eval | Stage-1 frozen encoder eval | 3 | 0.775956 +/- 0.007728 | 0.772940 +/- 0.007432 | 0.776559 +/- 0.007413 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 3 | ShuffleFAC encoder | ShipsEar | recording_eval | Stage-1 frozen encoder eval | 3 | 0.701755 +/- 0.065643 | 0.625507 +/- 0.081863 | 0.655520 +/- 0.079463 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 4 | ShuffleFAC encoder | ShipsEar | aggregation_eval | Stage-1 frozen encoder eval | 3 | 0.701755 +/- 0.065643 | 0.625507 +/- 0.081863 | 0.655520 +/- 0.079463 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 5 | GraphHead | DeepShip | attention | Stage-2 trained head | 3 | 0.775956 +/- 0.036860 | 0.771075 +/- 0.043832 | 0.775977 +/- 0.037857 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 6 | GraphHead | DeepShip | graph | Stage-2 trained head | 3 | 0.773224 +/- 0.020446 | 0.770719 +/- 0.023670 | 0.773592 +/- 0.021351 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 7 | GraphHead | DeepShip | graph_aware_attention | Stage-2 trained head | 3 | 0.781421 +/- 0.010223 | 0.772453 +/- 0.014504 | 0.780935 +/- 0.011098 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 8 | GraphHead | ShipsEar | attention | Stage-2 trained head | 3 | 0.736842 +/- 0.042974 | 0.687619 +/- 0.068641 | 0.706391 +/- 0.054167 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 9 | GraphHead | ShipsEar | graph | Stage-2 trained head | 3 | 0.684211 +/- 0.042974 | 0.635628 +/- 0.063697 | 0.652954 +/- 0.053170 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 10 | GraphHead | ShipsEar | graph_aware_attention | Stage-2 trained head | 3 | 0.719298 +/- 0.065643 | 0.676508 +/- 0.083986 | 0.694695 +/- 0.069825 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 11 | Signal-Noise decoupled | ShipsEar | SN baseline | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 12 | Signal-Noise decoupled | ShipsEar | SN threshold graph | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 13 | Signal-Noise decoupled | ShipsEar | SN temperature | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 14 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 15 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | Stage-2 trained downstream | 3 | 0.701754 +/- 0.049622 | 0.649841 +/- 0.070331 | 0.669591 +/- 0.065163 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 16 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | Stage-2 trained downstream | 3 | 0.701754 +/- 0.024811 | 0.644615 +/- 0.044065 | 0.667521 +/- 0.035805 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 17 | Signal-Noise decoupled | DeepShip | SN baseline | Stage-2 trained downstream | 3 | 0.775956 +/- 0.007728 | 0.779129 +/- 0.002623 | 0.776148 +/- 0.006917 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg graph degree 2.980; avg signal top-k 8.0 |
| 18 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | Stage-2 trained downstream | 3 | 0.784153 +/- 0.010223 | 0.786350 +/- 0.017095 | 0.784107 +/- 0.010699 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 2.981; avg signal top-k 4.0 |
| 19 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | Stage-2 trained downstream | 3 | 0.775956 +/- 0.007728 | 0.779129 +/- 0.002623 | 0.776148 +/- 0.006917 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, full soft attention; avg graph degree 6.490 |
| 20 | Signal-Noise decoupled | DeepShip | SN ETA Complete | Stage-2 trained downstream | 3 | 0.784153 +/- 0.010223 | 0.786350 +/- 0.017095 | 0.784107 +/- 0.010699 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=0; avg graph degree 6.134 |
| 21 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | Stage-2 trained downstream | 3 | 0.792350 +/- 0.007728 | 0.800943 +/- 0.017295 | 0.792980 +/- 0.008654 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 6.310 |

## 表 2：不同 Seed 明细指标

该表保留每个实验在不同 seed 上的 ACC、Macro-F1、Weighted-F1，方便观察随机种子波动。

| 序号 | 实验组 | 数据集 | 实验/模型 | Seed | ACC | Macro-F1 | Weighted-F1 | Params | Trainable Params | MACs | 备注 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ShuffleFAC encoder | DeepShip | recording_eval | 42 | 0.770492 | 0.762786 | 0.770803 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 2 | ShuffleFAC encoder | DeepShip | recording_eval | 43 | 0.786885 | 0.780370 | 0.787026 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 3 | ShuffleFAC encoder | DeepShip | recording_eval | 44 | 0.770492 | 0.775663 | 0.771849 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 4 | ShuffleFAC encoder | DeepShip | aggregation_eval | 42 | 0.770492 | 0.762786 | 0.770803 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 5 | ShuffleFAC encoder | DeepShip | aggregation_eval | 43 | 0.786885 | 0.780370 | 0.787026 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 6 | ShuffleFAC encoder | DeepShip | aggregation_eval | 44 | 0.770492 | 0.775663 | 0.771849 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 7 | ShuffleFAC encoder | ShipsEar | recording_eval | 42 | 0.631579 | 0.550476 | 0.577444 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 8 | ShuffleFAC encoder | ShipsEar | recording_eval | 43 | 0.789474 | 0.739377 | 0.764556 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 9 | ShuffleFAC encoder | ShipsEar | recording_eval | 44 | 0.684211 | 0.586667 | 0.624561 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 10 | ShuffleFAC encoder | ShipsEar | aggregation_eval | 42 | 0.631579 | 0.550476 | 0.577444 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 11 | ShuffleFAC encoder | ShipsEar | aggregation_eval | 43 | 0.789474 | 0.739377 | 0.764556 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 12 | ShuffleFAC encoder | ShipsEar | aggregation_eval | 44 | 0.684211 | 0.586667 | 0.624561 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 13 | GraphHead | DeepShip | attention | 42 | 0.827869 | 0.832826 | 0.829202 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 14 | GraphHead | DeepShip | attention | 43 | 0.754098 | 0.744891 | 0.754367 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 15 | GraphHead | DeepShip | attention | 44 | 0.745902 | 0.735507 | 0.744360 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 16 | GraphHead | DeepShip | graph | 42 | 0.795082 | 0.801320 | 0.797261 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 17 | GraphHead | DeepShip | graph | 43 | 0.778689 | 0.767170 | 0.777994 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 18 | GraphHead | DeepShip | graph | 44 | 0.745902 | 0.743667 | 0.745521 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 19 | GraphHead | DeepShip | graph_aware_attention | 42 | 0.795082 | 0.789561 | 0.795714 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 20 | GraphHead | DeepShip | graph_aware_attention | 43 | 0.770492 | 0.754099 | 0.768971 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 21 | GraphHead | DeepShip | graph_aware_attention | 44 | 0.778689 | 0.773699 | 0.778120 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 22 | GraphHead | ShipsEar | attention | 42 | 0.684211 | 0.593333 | 0.635088 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 23 | GraphHead | ShipsEar | attention | 43 | 0.789474 | 0.754762 | 0.766291 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 24 | GraphHead | ShipsEar | attention | 44 | 0.736842 | 0.714762 | 0.717794 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 25 | GraphHead | ShipsEar | graph | 42 | 0.631579 | 0.558788 | 0.587560 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 26 | GraphHead | ShipsEar | graph | 43 | 0.684211 | 0.633333 | 0.653509 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 27 | GraphHead | ShipsEar | graph | 44 | 0.736842 | 0.714762 | 0.717794 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 28 | GraphHead | ShipsEar | graph_aware_attention | 42 | 0.631579 | 0.560000 | 0.600000 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 29 | GraphHead | ShipsEar | graph_aware_attention | 43 | 0.789474 | 0.754762 | 0.766291 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 30 | GraphHead | ShipsEar | graph_aware_attention | 44 | 0.736842 | 0.714762 | 0.717794 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 31 | Signal-Noise decoupled | ShipsEar | SN baseline | 42 | 0.631579 | 0.560000 | 0.582456 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 32 | Signal-Noise decoupled | ShipsEar | SN baseline | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 33 | Signal-Noise decoupled | ShipsEar | SN baseline | 44 | 0.789474 | 0.777143 | 0.778947 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 34 | Signal-Noise decoupled | ShipsEar | SN threshold graph | 42 | 0.631579 | 0.560000 | 0.582456 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 35 | Signal-Noise decoupled | ShipsEar | SN threshold graph | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 36 | Signal-Noise decoupled | ShipsEar | SN threshold graph | 44 | 0.789474 | 0.777143 | 0.778947 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 37 | Signal-Noise decoupled | ShipsEar | SN temperature | 42 | 0.631579 | 0.560000 | 0.582456 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 38 | Signal-Noise decoupled | ShipsEar | SN temperature | 43 | 0.736842 | 0.703333 | 0.716667 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 39 | Signal-Noise decoupled | ShipsEar | SN temperature | 44 | 0.789474 | 0.777143 | 0.778947 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 40 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | 42 | 0.631579 | 0.560000 | 0.582456 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 41 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | 43 | 0.736842 | 0.703333 | 0.716667 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 42 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | 44 | 0.789474 | 0.777143 | 0.778947 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 43 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | 42 | 0.631579 | 0.550476 | 0.577444 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 44 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 45 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | 44 | 0.736842 | 0.695714 | 0.714662 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 46 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | 42 | 0.684211 | 0.597179 | 0.632389 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 47 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 48 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | 44 | 0.684211 | 0.633333 | 0.653509 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 49 | Signal-Noise decoupled | DeepShip | SN baseline | 42 | 0.770492 | 0.782781 | 0.772036 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg signal top-k 8.0 |
| 50 | Signal-Noise decoupled | DeepShip | SN baseline | 43 | 0.786885 | 0.777864 | 0.785891 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg signal top-k 8.0 |
| 51 | Signal-Noise decoupled | DeepShip | SN baseline | 44 | 0.770492 | 0.776741 | 0.770518 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg signal top-k 8.0 |
| 52 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | 42 | 0.786885 | 0.803874 | 0.787415 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg signal top-k 4.0 |
| 53 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | 43 | 0.795082 | 0.792011 | 0.795239 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg signal top-k 4.0 |
| 54 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | 44 | 0.770492 | 0.763165 | 0.769665 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg signal top-k 4.0 |
| 55 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | 42 | 0.770492 | 0.782781 | 0.772036 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 7.000 |
| 56 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | 43 | 0.786885 | 0.777864 | 0.785891 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 5.720 |
| 57 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | 44 | 0.770492 | 0.776741 | 0.770518 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.750 |
| 58 | Signal-Noise decoupled | DeepShip | SN ETA Complete | 42 | 0.786885 | 0.803874 | 0.787415 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4; avg graph degree 6.666 |
| 59 | Signal-Noise decoupled | DeepShip | SN ETA Complete | 43 | 0.795082 | 0.792011 | 0.795239 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4; avg graph degree 5.350 |
| 60 | Signal-Noise decoupled | DeepShip | SN ETA Complete | 44 | 0.770492 | 0.763165 | 0.769665 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4; avg graph degree 6.386 |
| 61 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | 42 | 0.803279 | 0.823044 | 0.805217 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 6.536 |
| 62 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | 43 | 0.786885 | 0.780818 | 0.786910 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 5.395 |
| 63 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | 44 | 0.786885 | 0.798967 | 0.786811 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 6.999 |

## 关键观察

- DeepShip 上，GraphHead 的 graph_aware_attention 在三 seed 平均 Macro-F1 为 0.772453，和 Stage-1 recording eval 的 0.772940 非常接近，提升不明显。
- ShipsEar 上，GraphHead graph_aware_attention 三 seed 平均 Macro-F1 为 0.676508，高于 ShuffleFAC Stage-1 recording eval 的 0.625507。
- ShipsEar 上，Signal-Noise baseline 三 seed 平均 Macro-F1 为 0.680159，略高于 GraphHead graph_aware_attention 的 0.676508，但差距很小。
- threshold graph、temperature、threshold+temperature 三个消融在 ShipsEar 上与 Signal-Noise baseline 指标完全持平。
- ExpD Top4 Warmup5 在 ShipsEar 上平均 Macro-F1 为 0.649841，低于 Signal-Noise baseline 的 0.680159，说明 Top-K sparse signal pooling 当前设置没有带来收益。
- ExpE Top4 NoWarmup 在 ShipsEar 上平均 Macro-F1 为 0.644615，也低于 Signal-Noise baseline，并且略低于 ExpD；纯 hard Top-K 从第 1 个 epoch 开始启用后没有带来收益。
- DeepShip 上，Signal-Noise baseline 三 seed 平均 Macro-F1 为 0.779129，略高于 Stage-1 recording eval 的 0.772940；ExpE Top4 NoWarmup 进一步到 0.786350，但 seed 间方差也更大。
- DeepShip 新增阈值图消融中，`sim_threshold=0.8` 的平均图度数为 6.13-6.49，明显高于目标健康稀疏区间 1.5-3.5，接近 8 clips 下的全连接图；其中 ExpD Warmup5 的 Macro-F1 最高，为 0.800943。

## 文件来源

- Stage-1 ShuffleFAC：`results/ShuffleFAC/*_recording_eval/seed_*/metrics.txt` 与 `results/ShuffleFAC/*_aggregation_eval/seed_*/metrics.txt`。
- GraphHead：`results/ShuffleFAC_GRAPHHEAD/0502_*_seed*_S8_ms5/metrics.txt`。
- Signal-Noise：`results/ShuffleFAC_SIGNAL_NOISE/*/seed*/metrics.json`。
- ExpD：`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_ExpD_Top4_Warmup5_Sim08/seed*/metrics.json`。
- ExpE：`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_ExpE_Top4_NoWarmup_TempSim/seed*/metrics.json`。
- DeepShip Signal-Noise：`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_Baseline_TempSim/seed*/metrics.json` 与 `results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ExpE_Top4_NoWarmup/seed*/metrics.json`。
- DeepShip threshold ablations：`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ThresholdOnly/seed*/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ETA_Complete/seed*/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ExpD_Warmup5/seed*/metrics.json`。
