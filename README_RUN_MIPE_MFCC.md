# MIPE + MFCC run guide

This guide runs the pipeline in `external/mipe+mfcc/mfcc+mipe/` from raw ShipsEar wav files to saved features, grouped cross-validation, metrics, and model weights.

## Install dependencies

Use the Python environment that has PyTorch installed. A minimal install is:

```bash
python -m pip install numpy scipy scikit-learn librosa==0.9.2 soundfile tqdm matplotlib torch torchvision
```

Optional FLOPs statistics:

```bash
python -m pip install thop
```

If your PyTorch/CUDA version is special, install `torch` and `torchvision` from the official PyTorch command first, then install the remaining packages above.

## Expected data layout

`--data_root` should point to the ShipsEar directory containing class folders:

```text
ShipsEar/
  ClassA/*.wav
  ClassB/*.wav
  ClassC/*.wav
  ClassD/*.wav
  ClassE/*.wav
```

The extractor also accepts ShipsEar folders named `A/`, `B/`, `C/`, `D/`, `E/`.
Nested segment layouts are supported too:

```text
shipsEar_AUDIOS/
  A/15__10_07_13_radaUno_Pasa/*Segment_*.wav
  B/<recording_name>/*Segment_*.wav
  ...
```

You can pass `--data_root` directly or set environment variable `DATA_ROOT`. If neither exists, feature extraction stops with a clear error.

## Feature extraction

```bash
python external/mipe+mfcc/mfcc+mipe/data_augmentation.py \
  --data_root /path/to/ShipsEar \
  --out_dir ./outputs/mipe_mfcc \
  --seg_sec 5 \
  --augment_per_seg 5
```

The extractor saves all generated files under `--out_dir`, including:

- `mfcc_augmented.npy`
- `mipe_augmented.npy`
- `labels_augmented.npy`
- `groups.npy`
- `segment_ids.npy`
- `sample_manifest.csv`
- `feature_config.json`
- `feature_extraction.log`

`groups.npy` is recording-level: every segment and augmentation from the same immediate wav shares one group id. For nested `A/<recording_name>/*Segment_*.wav` layouts, all segment wavs under the same `<recording_name>` directory share one group id. This avoids recording leakage across folds.

## Quick smoke test

This uses one wav per class and one augmentation per segment:

```bash
python external/mipe+mfcc/mfcc+mipe/data_augmentation.py \
  --data_root /path/to/ShipsEar \
  --out_dir ./outputs/mipe_mfcc_debug \
  --max_files_per_class 1 \
  --augment_per_seg 1
```

```bash
python external/mipe+mfcc/mfcc+mipe/test2.py \
  --data_dir ./outputs/mipe_mfcc_debug \
  --epochs 1 \
  --batch_size 4 \
  --n_splits 2
```

For CPU-only smoke testing, add:

```bash
--device cpu
```

## Full training

```bash
python external/mipe+mfcc/mfcc+mipe/test2.py \
  --data_dir ./outputs/mipe_mfcc \
  --epochs 150 \
  --batch_size 16 \
  --patience 20 \
  --n_splits 5
```

Training outputs are saved to `--data_dir` by default. You can separate them with `--out_dir ./outputs/mipe_mfcc_train`.

Expected training outputs:

- `best_model_fold1.pth`, `best_model_fold2.pth`, ...
- `metrics.csv`
- `results.json`
- `confusion_matrix_fold*.png`
- `confusion_matrix_all_folds.png`
- `train.log`

Each fold reports:

- ACC
- Macro-F1
- Weighted-F1
- Precision weighted
- Recall weighted

`metrics.csv` and `results.json` also include mean and standard deviation across folds.

## Common errors

`Missing --data_root and DATA_ROOT is not set.`

Pass `--data_root /path/to/ShipsEar` or set `DATA_ROOT`.

`Required file not found: groups.npy`

Run `data_augmentation.py` again with the updated script. `test2.py` no longer guesses groups from the augmentation count.

`n_splits is larger than unique groups`

Lower `--n_splits` or extract more recordings. GroupKFold needs at least one recording-level group per split.

`No module named librosa`

Install dependencies in the Python environment used to run the command:

```bash
python -m pip install librosa==0.9.2 soundfile scipy scikit-learn tqdm matplotlib
```

`CUDA was requested but torch.cuda.is_available() is false.`

Use `--device cpu` or install a CUDA-enabled PyTorch build.

`thop is not installed; skipping FLOPs statistics.`

This is safe. Install `thop` only if you need FLOPs.
