import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

# Compatibility for older librosa/sklearn dependency stacks on newer numpy.
if "complex" not in np.__dict__:
    np.complex = np.complex128  # type: ignore[attr-defined]
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

try:
    from thop import clever_format, profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


DEFAULT_CLASS_NAMES = ["ClassA", "ClassB", "ClassC", "ClassD", "ClassE"]
TARGET_FRAMES = 100
N_MFCC = 13
MIPE_SCALES = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
FUSION_TYPE = "adaptive_gate"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and validate the MIPE+MFCC model with recording-level GroupKFold."
    )
    parser.add_argument(
        "--data_dir",
        default="./outputs/mipe_mfcc",
        help="Directory containing mfcc_augmented.npy, mipe_augmented.npy, labels_augmented.npy, groups.npy.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory for logs, model weights, and metrics. Defaults to --data_dir.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else args.data_dir
    if args.epochs <= 0:
        parser.error("--epochs must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be > 0.")
    if args.patience <= 0:
        parser.error("--patience must be > 0.")
    if args.n_splits < 2:
        parser.error("--n_splits must be >= 2.")
    return args


def setup_logging(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mipe_mfcc_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_config(data_dir):
    config_path = data_dir / "feature_config.json"
    if not config_path.is_file():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path):
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path}")
    return path


def load_features(data_dir):
    mfcc = np.load(require_file(data_dir / "mfcc_augmented.npy"))
    mipe = np.load(require_file(data_dir / "mipe_augmented.npy"))
    labels = np.load(require_file(data_dir / "labels_augmented.npy"))
    groups = np.load(require_file(data_dir / "groups.npy"))

    n = len(labels)
    if not (len(mfcc) == len(mipe) == len(groups) == n):
        raise SystemExit(
            "Feature length mismatch: "
            f"mfcc={len(mfcc)}, mipe={len(mipe)}, labels={len(labels)}, groups={len(groups)}"
        )
    if mfcc.ndim != 4:
        raise SystemExit(f"Expected MFCC shape (N,T,13,2) or (N,2,T,13), got {mfcc.shape}")
    if mfcc.shape[-1] == 2:
        mfcc = mfcc.transpose(0, 3, 1, 2)
    elif mfcc.shape[1] != 2:
        raise SystemExit(f"Expected MFCC channel dimension of 2, got {mfcc.shape}")
    if mipe.ndim != 2:
        raise SystemExit(f"Expected MIPE shape (N,scales), got {mipe.shape}")

    return (
        mfcc.astype(np.float32),
        mipe.astype(np.float32),
        labels.astype(np.int64),
        groups.astype(np.int64),
    )


def make_resnet18():
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


class AdaptiveGateFusion(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, feat1, feat2):
        concat_feat = torch.cat([feat1, feat2], dim=1)
        weight = self.gate(concat_feat)
        weight1 = weight[:, 0:1].expand_as(feat1)
        weight2 = weight[:, 1:2].expand_as(feat2)
        fused_feat = weight1 * feat1 + weight2 * feat2
        return fused_feat, weight


class MFPNet(nn.Module):
    def __init__(self, mipe_dim, num_classes=5, fusion="adaptive_gate"):
        super().__init__()
        resnet = make_resnet18()
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mfcc_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.mfcc_norm = nn.LayerNorm(512)
        self.mfcc_fc = nn.Linear(512, 512)

        self.mipe_embed = nn.Linear(1, 128)
        self.pos_encoder = nn.Parameter(torch.randn(1, mipe_dim, 128))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True,
        )
        self.mipe_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mipe_norm = nn.LayerNorm(128)
        self.mipe_fc = nn.Linear(128, 512)

        self.fusion_type = fusion
        if self.fusion_type == "adaptive_gate":
            self.fusion = AdaptiveGateFusion(feat_dim=512)
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
        elif self.fusion_type == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def forward(self, mfcc_input, mipe_input):
        x_m = self.mfcc_backbone(mfcc_input).flatten(1)
        x_m = self.mfcc_norm(x_m)
        x_m = self.mfcc_fc(x_m)

        x_p = mipe_input.unsqueeze(-1)
        x_p = self.mipe_embed(x_p)
        x_p = x_p + self.pos_encoder
        x_p = self.mipe_transformer(x_p)
        x_p = x_p.mean(dim=1)
        x_p = self.mipe_norm(x_p)
        x_p = self.mipe_fc(x_p)

        if self.fusion_type == "adaptive_gate":
            feat, _ = self.fusion(x_m, x_p)
            return self.classifier(feat)
        feat = torch.cat([x_m, x_p], dim=1)
        return self.classifier(feat)


def print_model_complexity(model, device, mfcc_shape, mipe_dim, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: total=%s trainable=%s", f"{total_params:,}", f"{trainable_params:,}")

    if not THOP_AVAILABLE:
        logger.info("thop is not installed; skipping FLOPs statistics.")
        return

    try:
        mfcc_dummy = torch.randn(1, *mfcc_shape, device=device)
        mipe_dummy = torch.randn(1, mipe_dim, device=device)
        flops, params = profile(model, inputs=(mfcc_dummy, mipe_dummy), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        logger.info("FLOPs(single sample): %s, params: %s", flops, params)
    except Exception as exc:
        logger.warning("Failed to compute FLOPs with thop; skipping. Reason: %s", exc)


def compute_metrics(y_true, y_pred):
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Precision weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for mfcc_b, mipe_b, y_b in loader:
            mfcc_b = mfcc_b.to(device, non_blocking=True)
            mipe_b = mipe_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            out = model(mfcc_b, mipe_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * y_b.size(0)
            total_count += y_b.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y_b.cpu().numpy())
    return total_loss / max(1, total_count), np.asarray(trues), np.asarray(preds)


def train_fold(
    fold,
    train_mfcc,
    train_mipe,
    train_y,
    val_mfcc,
    val_mipe,
    val_y,
    args,
    device,
    num_classes,
    logger,
):
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_mfcc, dtype=torch.float32),
        torch.tensor(train_mipe, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_mfcc, dtype=torch.float32),
        torch.tensor(val_mipe, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MFPNet(mipe_dim=train_mipe.shape[1], fusion=FUSION_TYPE, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    best_metrics = None
    best_preds = None
    best_trues = None
    patience_counter = 0
    best_path = args.out_dir / f"best_model_fold{fold}.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for mfcc_b, mipe_b, y_b in train_loader:
            mfcc_b = mfcc_b.to(device, non_blocking=True)
            mipe_b = mipe_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(mfcc_b, mipe_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_b.size(0)
            total_count += y_b.size(0)

        train_loss = total_loss / max(1, total_count)
        val_loss, trues, preds = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(trues, preds)
        scheduler.step(val_loss)

        if metrics["ACC"] > best_acc:
            best_acc = metrics["ACC"]
            best_epoch = epoch
            best_metrics = metrics
            best_preds = preds.copy()
            best_trues = trues.copy()
            patience_counter = 0
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "num_classes": num_classes,
                    "mipe_dim": train_mipe.shape[1],
                    "fusion_type": FUSION_TYPE,
                },
                best_path,
            )
        else:
            patience_counter += 1

        should_log = epoch == 1 or epoch == args.epochs or epoch % 10 == 0
        if should_log:
            logger.info(
                "Fold %d epoch %d/%d | train_loss=%.4f val_loss=%.4f ACC=%.4f best=%.4f",
                fold,
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                metrics["ACC"],
                best_acc,
            )

        if patience_counter >= args.patience:
            logger.info("Fold %d early stopping at epoch %d", fold, epoch)
            break

    result = {
        "fold": fold,
        "epoch": best_epoch,
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "model_path": str(best_path),
        **(best_metrics or {}),
    }
    return result, best_trues, best_preds


def save_confusion_matrix(y_true, y_pred, class_names, out_path, logger):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        labels = list(range(len(class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(out_path.stem)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not save confusion matrix %s: %s", out_path, exc)


def write_metrics_csv(rows, summary, out_path):
    metric_keys = ["ACC", "Macro-F1", "Weighted-F1", "Precision weighted", "Recall weighted"]
    fieldnames = ["fold", "epoch", "n_train", "n_val", *metric_keys]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        writer.writerow({"fold": "mean", **{key: summary["mean"][key] for key in metric_keys}})
        writer.writerow({"fold": "std", **{key: summary["std"][key] for key in metric_keys}})


def summarize(rows):
    metric_keys = ["ACC", "Macro-F1", "Weighted-F1", "Precision weighted", "Recall weighted"]
    return {
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in metric_keys},
        "std": {key: float(np.std([row[key] for row in rows])) for key in metric_keys},
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    logger = setup_logging(args.out_dir)
    device = resolve_device(args.device)

    logger.info("data_dir: %s", args.data_dir)
    logger.info("out_dir: %s", args.out_dir)
    logger.info(
        "epochs=%d batch_size=%d patience=%d n_splits=%d device=%s seed=%d",
        args.epochs,
        args.batch_size,
        args.patience,
        args.n_splits,
        device,
        args.seed,
    )

    mfcc_feats, mipe_seqs, labels, groups = load_features(args.data_dir)
    config = load_feature_config(args.data_dir)
    class_map = config.get("class_map") or {name: idx for idx, name in enumerate(DEFAULT_CLASS_NAMES)}
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    num_classes = max(len(class_names), int(labels.max()) + 1)
    if len(class_names) < num_classes:
        class_names = class_names + [f"Class{i}" for i in range(len(class_names), num_classes)]

    unique_groups = np.unique(groups)
    if len(unique_groups) < args.n_splits:
        raise SystemExit(
            f"n_splits={args.n_splits} is larger than unique groups={len(unique_groups)}. "
            "Lower --n_splits or extract more recordings."
        )

    logger.info(
        "Loaded MFCC=%s MIPE=%s labels=%s groups=%s unique_groups=%d",
        mfcc_feats.shape,
        mipe_seqs.shape,
        labels.shape,
        groups.shape,
        len(unique_groups),
    )
    logger.info("Class counts: %s", {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})

    temp_model = MFPNet(mipe_dim=mipe_seqs.shape[1], fusion=FUSION_TYPE, num_classes=num_classes).to(device)
    print_model_complexity(temp_model, device, tuple(mfcc_feats.shape[1:]), mipe_seqs.shape[1], logger)
    del temp_model

    gkf = GroupKFold(n_splits=args.n_splits)
    fold_rows = []
    all_trues = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(mfcc_feats, labels, groups=groups), start=1
    ):
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Group leakage detected in fold {fold}: {sorted(overlap)[:5]}")

        logger.info("===== Fold %d/%d =====", fold, args.n_splits)
        logger.info(
            "Fold %d groups: train=%d val=%d samples: train=%d val=%d",
            fold,
            len(train_groups),
            len(val_groups),
            len(train_idx),
            len(val_idx),
        )
        result, trues, preds = train_fold(
            fold,
            mfcc_feats[train_idx],
            mipe_seqs[train_idx],
            labels[train_idx],
            mfcc_feats[val_idx],
            mipe_seqs[val_idx],
            labels[val_idx],
            args,
            device,
            num_classes,
            logger,
        )
        fold_rows.append(result)
        all_trues.extend(trues.tolist())
        all_preds.extend(preds.tolist())
        save_confusion_matrix(
            trues,
            preds,
            class_names,
            args.out_dir / f"confusion_matrix_fold{fold}.png",
            logger,
        )
        logger.info(
            "Fold %d best | ACC=%.4f Macro-F1=%.4f Weighted-F1=%.4f Precision weighted=%.4f Recall weighted=%.4f",
            fold,
            result["ACC"],
            result["Macro-F1"],
            result["Weighted-F1"],
            result["Precision weighted"],
            result["Recall weighted"],
        )

    summary = summarize(fold_rows)
    save_confusion_matrix(
        np.asarray(all_trues),
        np.asarray(all_preds),
        class_names,
        args.out_dir / "confusion_matrix_all_folds.png",
        logger,
    )

    write_metrics_csv(fold_rows, summary, args.out_dir / "metrics.csv")
    results = {
        "args": {
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "device": str(device),
            "n_splits": args.n_splits,
            "seed": args.seed,
        },
        "class_names": class_names,
        "folds": fold_rows,
        "summary": summary,
    }
    with open(args.out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("===== Summary =====")
    for metric, value in summary["mean"].items():
        logger.info("%s: %.4f +/- %.4f", metric, value, summary["std"][metric])
    logger.info("Saved metrics.csv and results.json to %s", args.out_dir)


if __name__ == "__main__":
    main()
