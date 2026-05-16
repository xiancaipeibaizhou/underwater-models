#!/usr/bin/env python3
"""Evaluate trained Graph-aware AttentionHead artifacts.

This script is intentionally evaluation-only. It loads a trained external
ShuffleFAC encoder checkpoint plus a trained graph_aware_attention head, then
evaluates recording-level deterministic multi-bag predictions from existing
recording-level feature caches.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from run_graphhead import (
    FrozenShuffleFACEncoder,
    GraphAwareAttentionHead,
    RecordingBagCachedDataset,
    resolve_path,
)


DEEPSHIP_CLASS_NAMES = ["Cargo", "Passengership", "Tanker", "Tug"]
SHIPSEAR_CLASS_NAMES = ["A", "B", "C", "D", "E"]


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def path_or_none(value):
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def expand_path_arg(value: str, root: Path):
    """Expand comma-separated, glob, directory, or ordinary path arguments."""
    value = str(value)
    if value.lower() == "auto":
        return ["auto"]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    out = []
    for part in parts:
        if any(ch in part for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(part)]
            out.extend(matches)
            continue
        p = resolve_path(part, root)
        if p.is_dir():
            out.extend(sorted(p.rglob("*graph_aware_attention*/best_head.pt")))
        else:
            out.append(p)
    return sorted(out, key=lambda p: str(p)) if out and out != ["auto"] else out


def infer_model_config_from_head(head_path: Path) -> Path:
    candidate = head_path.parent / "model_config.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not infer model_config next to {head_path}")
    return candidate


def infer_encoder_from_config(config: dict, root: Path) -> Path:
    value = config.get("Encoder checkpoint") or config.get("encoder_ckpt")
    if not value:
        training = config.get("training", {})
        value = training.get("ckpt_path")
    if not value:
        raise ValueError("Could not infer encoder checkpoint from model_config")
    return resolve_path(value, root)


def pair_runs(args, root: Path):
    head_paths = expand_path_arg(args.graph_head_ckpt, root)
    if head_paths == ["auto"]:
        raise ValueError("--graph_head_ckpt cannot be auto; pass a file, glob, comma list, or directory")
    if not head_paths:
        raise FileNotFoundError(f"No graph head checkpoints matched: {args.graph_head_ckpt}")

    encoder_paths = expand_path_arg(args.encoder_ckpt, root)
    config_paths = expand_path_arg(args.model_config, root)

    runs = []
    for index, head_path in enumerate(head_paths):
        if not head_path.exists():
            raise FileNotFoundError(f"graph_head_ckpt not found: {head_path}")

        if config_paths == ["auto"]:
            config_path = infer_model_config_from_head(head_path)
        elif len(config_paths) == 1:
            config_path = config_paths[0]
        elif len(config_paths) == len(head_paths):
            config_path = config_paths[index]
        else:
            raise ValueError("--model_config count must be 1, auto, or match graph_head_ckpt count")

        config = load_json(config_path)

        if encoder_paths == ["auto"]:
            encoder_path = infer_encoder_from_config(config, root)
        elif len(encoder_paths) == 1:
            encoder_path = encoder_paths[0]
        elif len(encoder_paths) == len(head_paths):
            encoder_path = encoder_paths[index]
        else:
            raise ValueError("--encoder_ckpt count must be 1, auto, or match graph_head_ckpt count")

        if not encoder_path.exists():
            raise FileNotFoundError(f"encoder_ckpt not found: {encoder_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"model_config not found: {config_path}")
        runs.append(
            {
                "encoder_ckpt": encoder_path,
                "graph_head_ckpt": head_path,
                "model_config": config_path,
                "model_config_payload": config,
            }
        )
    return runs


def pick(value, checkpoint_args: dict, name: str, default):
    if value is not None:
        return value
    if isinstance(checkpoint_args, dict) and checkpoint_args.get(name) is not None:
        return checkpoint_args[name]
    return default


def checkpoint_args(checkpoint: dict):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def normalize_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state")
        if state is None:
            state = checkpoint.get("state_dict")
        if state is None:
            state = checkpoint
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("Graph head checkpoint does not contain a state dict")
    return {str(k).removeprefix("module."): v for k, v in state.items()}


def prefixed(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name


def load_graphaware_model(
    encoder_ckpt: Path,
    graph_head_ckpt: Path,
    device: torch.device,
    graph_k: int,
    edge_mode: str,
    dropout: float,
    output_dir: Path,
    output_prefix: str = "",
):
    encoder = FrozenShuffleFACEncoder(encoder_ckpt, device=device)
    for param in encoder.parameters():
        param.requires_grad = False

    num_classes = int(encoder.cnn_cfg["n_class"])
    model = GraphAwareAttentionHead(
        encoder=encoder,
        num_classes=num_classes,
        graph_k=int(graph_k),
        edge_mode=str(edge_mode),
        dropout=float(dropout),
    ).to(device)

    checkpoint = torch_load(graph_head_ckpt, map_location=device)
    state = normalize_state_dict(checkpoint)
    warning_path = output_dir / prefixed(output_prefix, "load_warnings.txt")
    try:
        model.load_state_dict(state, strict=True)
        warning_path.write_text("strict=True load succeeded.\n", encoding="utf-8")
    except RuntimeError as exc:
        result = model.load_state_dict(state, strict=False)
        warning_path.write_text(
            "strict=True load failed; strict=False fallback was used.\n\n"
            f"strict_error:\n{exc}\n\n"
            f"missing_keys:\n{list(result.missing_keys)}\n\n"
            f"unexpected_keys:\n{list(result.unexpected_keys)}\n",
            encoding="utf-8",
        )

    model.eval()
    model.encoder.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    return model, checkpoint


def validate_cache(cache_path: Path, split_name: str):
    payload = torch_load(cache_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    protocol = metadata.get("protocol")
    if protocol != "recording_level":
        raise ValueError(
            f"Cache {cache_path} is not strict recording-level. "
            f"metadata.protocol={protocol!r}"
        )
    actual_split = payload.get("split_name")
    if actual_split and actual_split != split_name:
        raise ValueError(f"Cache split mismatch for {cache_path}: expected {split_name}, got {actual_split}")
    return payload


def resolve_cache_paths(config: dict, root: Path) -> dict:
    cache_paths = config.get("cache_paths")
    if not isinstance(cache_paths, dict):
        raise ValueError("model_config must contain cache_paths")
    for key in ["train", "val", "test"]:
        if key not in cache_paths:
            raise ValueError(f"model_config cache_paths is missing {key}")
    return {key: resolve_path(value, root) for key, value in cache_paths.items()}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@torch.no_grad()
def collect_multisample_predictions(model, dataset, batch_size, criterion, device, eval_samples: int):
    model.eval()
    model.encoder.encoder.eval()

    losses = []
    total = 0
    y_true = []
    logits_all = []
    recording_ids = []
    entropy_vals = []
    delta_vals = []

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        labels = torch.tensor(
            [int(dataset.recordings[i][2]) for i in indices],
            dtype=torch.long,
            device=device,
        )
        rids = [str(dataset.recordings[i][0]) for i in indices]

        sample_logits = []
        sample_entropy = []
        sample_delta = []
        for sample_id in range(eval_samples):
            clips = torch.stack(
                [dataset.get_eval_item(i, sample_id, eval_samples)[0] for i in indices],
                dim=0,
            ).to(device, non_blocking=True)
            logits = model(clips)
            sample_logits.append(logits.detach())
            sample_entropy.append(float(model.last_attn_entropy.detach().cpu()))
            sample_delta.append(float(model.last_graph_delta_norm.detach().cpu()))

        logits = torch.stack(sample_logits, dim=0).mean(dim=0)
        loss = criterion(logits, labels)
        batch_size_actual = int(labels.size(0))
        losses.append(float(loss.detach().cpu()) * batch_size_actual)
        total += batch_size_actual

        y_true.extend(labels.detach().cpu().numpy().tolist())
        logits_all.append(logits.detach().cpu())
        recording_ids.extend(rids)
        entropy_vals.append(float(np.mean(sample_entropy)))
        delta_vals.append(float(np.mean(sample_delta)))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    logits_np = torch.cat(logits_all, dim=0).numpy()
    prob_np = softmax_np(logits_np)
    pred_np = logits_np.argmax(axis=1).astype(np.int64)

    metrics = {
        "y_true": y_true_np,
        "y_pred": pred_np,
        "y_logits": logits_np,
        "y_prob": prob_np,
        "recording_ids": recording_ids,
        "loss": float(sum(losses) / max(total, 1)),
        "ACC": float(accuracy_score(y_true_np, pred_np)),
        "Macro-F1": float(f1_score(y_true_np, pred_np, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true_np, pred_np, average="weighted", zero_division=0)),
        "Precision macro": float(precision_score(y_true_np, pred_np, average="macro", zero_division=0)),
        "Precision weighted": float(precision_score(y_true_np, pred_np, average="weighted", zero_division=0)),
        "Recall macro": float(recall_score(y_true_np, pred_np, average="macro", zero_division=0)),
        "Recall weighted": float(recall_score(y_true_np, pred_np, average="weighted", zero_division=0)),
        "attn_entropy": float(np.mean(entropy_vals)) if entropy_vals else math.nan,
        "graph_delta_norm": float(np.mean(delta_vals)) if delta_vals else math.nan,
        "graph_res_scale": float(model.graph_res_scale.detach().cpu()),
    }
    return metrics


def infer_dataset_name(dataset_arg: str, config: dict, encoder_ckpt: Path, num_classes: int):
    if dataset_arg != "auto":
        return dataset_arg
    for key in ["dataset", "Dataset"]:
        value = config.get(key)
        if value in {"DeepShip", "ShipsEar"}:
            return value
    split_meta = config.get("split_metadata", {})
    value = split_meta.get("dataset")
    if value in {"DeepShip", "ShipsEar"}:
        return value
    text = str(encoder_ckpt) + " " + str(config.get("Encoder checkpoint", ""))
    if "ShipsEar" in text:
        return "ShipsEar"
    if "DeepShip" in text:
        return "DeepShip"
    if num_classes == 4:
        return "DeepShip"
    if num_classes == 5:
        return "ShipsEar"
    return "auto"


def class_names_for(dataset_name: str, num_classes: int):
    if dataset_name == "DeepShip" and num_classes == 4:
        return DEEPSHIP_CLASS_NAMES
    if dataset_name == "ShipsEar" and num_classes == 5:
        return SHIPSEAR_CLASS_NAMES
    return [f"class_{idx}" for idx in range(num_classes)]


def write_matrix_csv(path: Path, matrix: np.ndarray, class_names: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for idx, row in enumerate(matrix):
            writer.writerow([class_names[idx], *row.tolist()])


def write_count_ratio_csv(path: Path, counts: np.ndarray, ratios: np.ndarray, class_names: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for i, row in enumerate(counts):
            formatted = []
            for j, value in enumerate(row):
                ratio = float(ratios[i, j])
                if int(value) == 0 or ratio == 0.0:
                    formatted.append("0")
                else:
                    formatted.append(f"{int(value)} ({ratio * 100.0:.2f}%)")
            writer.writerow([class_names[i], *formatted])


def plot_confusion(path: Path, matrix: np.ndarray, class_names: list[str], normalized: bool):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.4), max(5, len(class_names) * 1.2)))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    title = "Graph-aware AttentionHead Recording-level Confusion Matrix"
    if normalized:
        title += " (Row-normalized)"
    else:
        title += " (Counts)"
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_confusion_count_ratio(path: Path, counts: np.ndarray, ratios: np.ndarray, class_names: list[str]):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.5), max(5, len(class_names) * 1.25)))
    im = ax.imshow(ratios, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row percentage")
    ax.set_title("Graph-aware AttentionHead Recording-level Confusion Matrix\nCounts and Row Percentages")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            count = int(counts[i, j])
            ratio = float(ratios[i, j])
            if count == 0 or ratio == 0.0:
                text = "0"
            else:
                text = f"{count}\n{ratio * 100.0:.1f}%"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if ratio > 0.5 else "black",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_confusion_outputs(out_dir: Path, y_true, y_pred, class_names, output_prefix: str = ""):
    labels = list(range(len(class_names)))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = counts.astype(np.float64) / counts.sum(axis=1, keepdims=True)
    normalized = np.nan_to_num(normalized)

    write_count_ratio_csv(
        out_dir / prefixed(output_prefix, "confusion_matrix_counts_ratio.csv"),
        counts,
        normalized,
        class_names,
    )
    plot_confusion_count_ratio(
        out_dir / prefixed(output_prefix, "confusion_matrix_counts_ratio.png"),
        counts,
        normalized,
        class_names,
    )
    if not output_prefix:
        write_matrix_csv(out_dir / "confusion_matrix_counts.csv", counts, class_names)
        write_matrix_csv(out_dir / "confusion_matrix_normalized.csv", normalized, class_names)
        plot_confusion(out_dir / "confusion_matrix_counts.png", counts, class_names, normalized=False)
        plot_confusion(out_dir / "confusion_matrix_normalized.png", normalized, class_names, normalized=True)
    return counts, normalized


def save_predictions(out_dir: Path, metrics: dict, class_names: list[str], output_prefix: str = ""):
    np.save(out_dir / prefixed(output_prefix, "y_true.npy"), metrics["y_true"])
    np.save(out_dir / prefixed(output_prefix, "y_pred.npy"), metrics["y_pred"])
    np.save(out_dir / prefixed(output_prefix, "y_logits.npy"), metrics["y_logits"])
    np.save(out_dir / prefixed(output_prefix, "y_prob.npy"), metrics["y_prob"])

    fieldnames = [
        "recording_id",
        "true_label",
        "pred_label",
        "true_name",
        "pred_name",
        "confidence",
    ]
    fieldnames += [f"logit_class_{idx}" for idx in range(len(class_names))]
    fieldnames += [f"prob_class_{idx}" for idx in range(len(class_names))]

    with (out_dir / prefixed(output_prefix, "predictions_recording_level.csv")).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, rid in enumerate(metrics["recording_ids"]):
            true_label = int(metrics["y_true"][idx])
            pred_label = int(metrics["y_pred"][idx])
            prob = metrics["y_prob"][idx]
            logits = metrics["y_logits"][idx]
            row = {
                "recording_id": rid,
                "true_label": true_label,
                "pred_label": pred_label,
                "true_name": class_names[true_label] if true_label < len(class_names) else str(true_label),
                "pred_name": class_names[pred_label] if pred_label < len(class_names) else str(pred_label),
                "confidence": float(np.max(prob)),
            }
            for class_idx in range(len(class_names)):
                row[f"logit_class_{class_idx}"] = float(logits[class_idx])
                row[f"prob_class_{class_idx}"] = float(prob[class_idx])
            writer.writerow(row)

    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=6,
        zero_division=0,
    )
    (out_dir / prefixed(output_prefix, "classification_report.txt")).write_text(report, encoding="utf-8")


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def save_loss_curves(out_dir: Path, loss_log: Optional[Path], output_prefix: str = ""):
    compact_outputs = bool(output_prefix)
    if loss_log is None or not loss_log.exists():
        (out_dir / prefixed(output_prefix, "loss_curve_missing.txt")).write_text(
            "loss curve source not found. No epoch_metrics.csv was found, so only final eval loss is available.\n",
            encoding="utf-8",
        )
        return {"loss_curve_available": False, "loss_log": str(loss_log) if loss_log else None}

    rows = read_csv_rows(loss_log)
    if not rows:
        (out_dir / prefixed(output_prefix, "loss_curve_missing.txt")).write_text(
            f"loss curve source not found. Empty log: {loss_log}\n",
            encoding="utf-8",
        )
        return {"loss_curve_available": False, "loss_log": str(loss_log)}

    with (out_dir / prefixed(output_prefix, "loss_curve_graphaware.csv")).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    epoch = np.asarray([to_float(row.get("epoch")) for row in rows], dtype=np.float64)
    train_loss = np.asarray([to_float(row.get("train_loss")) for row in rows], dtype=np.float64)
    val_loss = np.asarray([to_float(row.get("val_loss")) for row in rows], dtype=np.float64)

    if not np.all(np.isnan(train_loss)) or not np.all(np.isnan(val_loss)):
        fig, ax = plt.subplots(figsize=(7, 5))
        if not np.all(np.isnan(train_loss)):
            ax.plot(epoch, train_loss, label="train_loss", linewidth=2)
        if not np.all(np.isnan(val_loss)):
            ax.plot(epoch, val_loss, label="val_loss", linewidth=2)
        ax.set_title("Graph-aware AttentionHead Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / prefixed(output_prefix, "loss_trainloss_valloss.png"), dpi=220)
        if not compact_outputs:
            fig.savefig(out_dir / "loss_curve_graphaware.png", dpi=220)
        plt.close(fig)

    val_macro_f1 = np.asarray([to_float(row.get("val_macro_f1")) for row in rows], dtype=np.float64)
    if not compact_outputs and not np.all(np.isnan(val_macro_f1)):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epoch, val_macro_f1, label="val_macro_f1", linewidth=2)
        ax.set_title("Graph-aware AttentionHead Val Macro-F1 Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Macro-F1")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / prefixed(output_prefix, "val_macro_f1_curve_graphaware.png"), dpi=220)
        plt.close(fig)

    if not np.all(np.isnan(train_loss)) and not np.all(np.isnan(val_macro_f1)):
        fig, ax_loss = plt.subplots(figsize=(7.4, 5))
        line_loss = ax_loss.plot(epoch, train_loss, label="train_loss", linewidth=2, color="#1f77b4")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Train Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.4)

        ax_f1 = ax_loss.twinx()
        line_f1 = ax_f1.plot(epoch, val_macro_f1, label="val_macro_f1", linewidth=2, color="#2ca02c")
        ax_f1.set_ylabel("Val Macro-F1")

        lines = line_loss + line_f1
        labels = [line.get_label() for line in lines]
        ax_loss.legend(lines, labels, loc="best")
        ax_loss.set_title("Graph-aware AttentionHead Train Loss and Val Macro-F1")
        fig.tight_layout()
        fig.savefig(out_dir / prefixed(output_prefix, "loss_trainloss_valf1.png"), dpi=220)
        plt.close(fig)

    return {"loss_curve_available": True, "loss_log": str(loss_log)}


def save_encoder_loss_curve(out_dir: Path, encoder_ckpt: Path, output_prefix: str = ""):
    train_log = encoder_ckpt.parent / "train_log.txt"
    if not train_log.exists():
        return None
    rows = read_csv_rows(train_log)
    if not rows:
        return None
    epoch = np.asarray([to_float(row.get("epoch")) for row in rows], dtype=np.float64)
    train_loss = np.asarray([to_float(row.get("train_loss")) for row in rows], dtype=np.float64)
    val_loss = np.asarray([to_float(row.get("val_loss")) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epoch, train_loss, label="encoder_train_loss", linewidth=2)
    ax.plot(epoch, val_loss, label="encoder_val_loss", linewidth=2)
    ax.set_title("ShuffleFAC Encoder Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / prefixed(output_prefix, "loss_curve_encoder.png"), dpi=220)
    plt.close(fig)
    return str(train_log)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = total - encoder_params
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "encoder_params": int(encoder_params),
        "head_params": int(head_params),
    }


def format_big_number(value: int):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def count_macs_hooks(model, dummy_input: torch.Tensor):
    hooks = []
    macs = {"total": 0}

    def conv_hook(module, inputs, output):
        x = inputs[0]
        out = output[0] if isinstance(output, tuple) else output
        if not isinstance(out, torch.Tensor) or out.ndim < 4:
            return
        batch_size = int(x.shape[0])
        out_channels = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        kernel_ops = int(module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups))
        macs["total"] += batch_size * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if not isinstance(out, torch.Tensor):
            return
        macs["total"] += int(out.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    training_state = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        model.train(training_state)
        for hook in hooks:
            hook.remove()
    return int(macs["total"])


def compute_complexity(model, dataset, device):
    try:
        clips, _label, _rid = dataset.get_eval_item(0, 0, 1)
        dummy = clips.unsqueeze(0).to(device)
        macs = count_macs_hooks(model, dummy)
        return {
            "macs_available": True,
            "input_shape": list(dummy.shape),
            "macs": int(macs),
            "macs_formatted": format_big_number(int(macs)),
            "macs_method": "conv_linear_forward_hooks",
        }
    except Exception as exc:
        return {
            "macs_available": False,
            "reason": str(exc),
            "macs": None,
            "macs_formatted": None,
            "macs_method": None,
        }


def save_param_outputs(out_dir: Path, param_summary: dict, complexity: dict, output_prefix: str = ""):
    write_json(out_dir / prefixed(output_prefix, "params_summary.json"), param_summary)
    write_json(out_dir / prefixed(output_prefix, "complexity_summary.json"), complexity)
    with (out_dir / prefixed(output_prefix, "params_summary.csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(param_summary.keys()))
        writer.writeheader()
        writer.writerow(param_summary)
    with (out_dir / prefixed(output_prefix, "params_summary.txt")).open("w", encoding="utf-8") as f:
        for key, value in param_summary.items():
            f.write(f"{key}: {value}\n")
        f.write(f"macs_available: {complexity.get('macs_available')}\n")
        f.write(f"macs: {complexity.get('macs')}\n")
        f.write(f"macs_formatted: {complexity.get('macs_formatted')}\n")
        f.write(f"macs_method: {complexity.get('macs_method')}\n")
        if not complexity.get("macs_available"):
            f.write(f"macs_reason: {complexity.get('reason')}\n")


def write_metrics_txt(path: Path, metrics: dict):
    with path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
            else:
                f.write(f"{key}: {value}\n")


def seed_from_path_or_checkpoint(path: Path, checkpoint: dict):
    args = checkpoint_args(checkpoint)
    if "seed" in args:
        return str(args["seed"])
    match = re.search(r"seed[_-]?(\d+)", str(path))
    return match.group(1) if match else "unknown"


def run_name(run: dict, checkpoint: dict, dataset_name: str):
    seed = seed_from_path_or_checkpoint(run["graph_head_ckpt"], checkpoint)
    parent = run["graph_head_ckpt"].parent.name
    name = f"{dataset_name}_seed{seed}_{parent}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def select_loss_log(cli_loss_log, graph_head_ckpt: Path, root: Path):
    value = path_or_none(cli_loss_log)
    if value is not None:
        return resolve_path(value, root)
    candidate = graph_head_ckpt.parent / "epoch_metrics.csv"
    return candidate if candidate.exists() else None


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def evaluate_one_run(run: dict, args, output_dir: Path, root: Path):
    output_prefix = "review_" if same_path(output_dir, run["graph_head_ckpt"].parent) else ""
    graph_ckpt_payload = torch_load(run["graph_head_ckpt"], map_location="cpu")
    graph_args = checkpoint_args(graph_ckpt_payload)
    clips_per_recording = int(pick(args.clips_per_recording, graph_args, "clips_per_recording", 8))
    eval_samples = int(pick(args.eval_samples, graph_args, "eval_samples", 5))
    batch_size = int(pick(args.batch_size, graph_args, "batch_size", 8))
    graph_k = int(pick(args.graph_k, graph_args, "graph_k", 2))
    edge_mode = str(pick(args.edge_mode, graph_args, "edge_mode", "temporal_similarity"))
    dropout = float(pick(args.dropout, graph_args, "dropout", 0.2))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, loaded_head = load_graphaware_model(
        encoder_ckpt=run["encoder_ckpt"],
        graph_head_ckpt=run["graph_head_ckpt"],
        device=device,
        graph_k=graph_k,
        edge_mode=edge_mode,
        dropout=dropout,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    num_classes = int(model.encoder.cnn_cfg["n_class"])
    dataset_name = infer_dataset_name(
        args.dataset,
        run["model_config_payload"],
        run["encoder_ckpt"],
        num_classes,
    )
    class_names = class_names_for(dataset_name, num_classes)

    cache_paths = resolve_cache_paths(run["model_config_payload"], root)
    split_names = ["val", "test"] if args.split == "both" else [args.split]
    criterion = nn.CrossEntropyLoss()

    param_summary = count_params(model)
    per_split_metrics = {}
    first_dataset = None
    split_dirs = {}
    for split_name in split_names:
        split_out = output_dir if len(split_names) == 1 else output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        validate_cache(cache_paths[split_name], split_name)
        dataset = RecordingBagCachedDataset(
            cache_path=cache_paths[split_name],
            clips_per_recording=clips_per_recording,
            train=False,
            seed=int(graph_args.get("seed", 42)),
        )
        if first_dataset is None:
            first_dataset = dataset

        pred_metrics = collect_multisample_predictions(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            criterion=criterion,
            device=device,
            eval_samples=eval_samples,
        )
        save_confusion_outputs(
            split_out,
            pred_metrics["y_true"],
            pred_metrics["y_pred"],
            class_names,
            output_prefix=output_prefix,
        )
        save_predictions(split_out, pred_metrics, class_names, output_prefix=output_prefix)

        prefix = "test" if split_name == "test" else split_name
        metric_payload = {
            "split": split_name,
            "head_type": "graph_aware_attention",
            "encoder_ckpt": str(run["encoder_ckpt"]),
            "graph_head_ckpt": str(run["graph_head_ckpt"]),
            "model_config": str(run["model_config"]),
            "dataset": dataset_name,
            "clips_per_recording": clips_per_recording,
            "eval_samples": eval_samples,
            "batch_size": batch_size,
            "graph_k": graph_k,
            "edge_mode": edge_mode,
            "dropout": dropout,
            f"{prefix}_loss": pred_metrics["loss"],
            f"{prefix}_acc": pred_metrics["ACC"],
            f"{prefix}_macro_f1": pred_metrics["Macro-F1"],
            f"{prefix}_weighted_f1": pred_metrics["Weighted-F1"],
            f"{prefix}_precision_macro": pred_metrics["Precision macro"],
            f"{prefix}_precision_weighted": pred_metrics["Precision weighted"],
            f"{prefix}_recall_macro": pred_metrics["Recall macro"],
            f"{prefix}_recall_weighted": pred_metrics["Recall weighted"],
            "attn_entropy": pred_metrics["attn_entropy"],
            "graph_delta_norm": pred_metrics["graph_delta_norm"],
            "graph_res_scale": pred_metrics["graph_res_scale"],
            **param_summary,
            "class_names": class_names,
        }
        per_split_metrics[split_name] = metric_payload
        split_dirs[split_name] = split_out

    complexity = compute_complexity(model, first_dataset, device) if first_dataset is not None else {
        "macs_available": False,
        "reason": "No dataset selected",
        "macs": None,
        "macs_formatted": None,
        "macs_method": None,
    }
    save_param_outputs(output_dir, param_summary, complexity, output_prefix=output_prefix)

    loss_log = select_loss_log(args.loss_log, run["graph_head_ckpt"], root)
    loss_info = save_loss_curves(output_dir, loss_log, output_prefix=output_prefix)
    encoder_loss_log = None
    if not output_prefix:
        encoder_loss_log = save_encoder_loss_curve(output_dir, run["encoder_ckpt"])

    used = [
        f"encoder_ckpt: {run['encoder_ckpt']}",
        f"graph_head_ckpt: {run['graph_head_ckpt']}",
        f"model_config: {run['model_config']}",
        f"loss_log: {loss_log if loss_log else 'loss curve source not found'}",
        f"encoder_loss_log: {encoder_loss_log if encoder_loss_log else 'not found'}",
    ]
    (output_dir / prefixed(output_prefix, "used_checkpoints.txt")).write_text(
        "\n".join(used) + "\n",
        encoding="utf-8",
    )

    final_payloads = {}
    for split_name, metric_payload in per_split_metrics.items():
        enriched = {
            **metric_payload,
            "macs": complexity.get("macs"),
            "macs_formatted": complexity.get("macs_formatted"),
            "macs_method": complexity.get("macs_method"),
            "macs_available": complexity.get("macs_available"),
            "loss_curve_available": loss_info.get("loss_curve_available"),
        }
        if "test_loss" not in enriched and split_name != "test":
            # Keep a stable generic view for non-test-only inspection.
            enriched["eval_loss"] = enriched.get(f"{split_name}_loss")
            enriched["eval_acc"] = enriched.get(f"{split_name}_acc")
            enriched["eval_macro_f1"] = enriched.get(f"{split_name}_macro_f1")
        target_dir = split_dirs[split_name]
        write_json(target_dir / prefixed(output_prefix, "metrics.json"), enriched)
        write_metrics_txt(target_dir / prefixed(output_prefix, "metrics.txt"), enriched)
        final_payloads[split_name] = enriched

    if len(final_payloads) > 1:
        write_json(output_dir / prefixed(output_prefix, "split_summary.json"), final_payloads)
        with (output_dir / prefixed(output_prefix, "split_summary.csv")).open("w", encoding="utf-8", newline="") as f:
            cols = [
                "split",
                "dataset",
                "test_acc",
                "test_macro_f1",
                "val_acc",
                "val_macro_f1",
                "attn_entropy",
                "graph_delta_norm",
                "graph_res_scale",
            ]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for payload in final_payloads.values():
                writer.writerow({col: payload.get(col, "") for col in cols})

    checkpoint_seed = seed_from_path_or_checkpoint(run["graph_head_ckpt"], loaded_head)
    summary_split = "test" if "test" in final_payloads else next(iter(final_payloads))
    summary_payload = final_payloads[summary_split]
    return {
        "run_name": output_dir.name,
        "seed": checkpoint_seed,
        "dataset": dataset_name,
        "split": summary_split,
        "encoder_ckpt": str(run["encoder_ckpt"]),
        "graph_head_ckpt": str(run["graph_head_ckpt"]),
        "metric_acc": summary_payload.get(f"{summary_split}_acc", summary_payload.get("test_acc")),
        "metric_macro_f1": summary_payload.get(f"{summary_split}_macro_f1", summary_payload.get("test_macro_f1")),
        "metric_weighted_f1": summary_payload.get(
            f"{summary_split}_weighted_f1",
            summary_payload.get("test_weighted_f1"),
        ),
        "attn_entropy": summary_payload.get("attn_entropy"),
        "graph_delta_norm": summary_payload.get("graph_delta_norm"),
        "graph_res_scale": summary_payload.get("graph_res_scale"),
        "total_params": param_summary["total_params"],
        "trainable_params": param_summary["trainable_params"],
        "macs": complexity.get("macs"),
        "macs_formatted": complexity.get("macs_formatted"),
        "loss_curve_available": loss_info.get("loss_curve_available"),
    }


def write_multiseed_summary(output_dir: Path, summaries: list[dict], output_prefix: str = ""):
    if len(summaries) <= 1:
        return
    write_json(output_dir / prefixed(output_prefix, "multiseed_summary.json"), {"runs": summaries})
    columns = list(summaries[0].keys())
    with (output_dir / prefixed(output_prefix, "multiseed_summary.csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Graph-aware AttentionHead artifacts.")
    parser.add_argument("--encoder_ckpt", required=True, help="ShuffleFAC encoder best.pt path, comma list, glob, or auto.")
    parser.add_argument("--graph_head_ckpt", required=True, help="Graph-aware best_head.pt path, comma list, glob, or directory.")
    parser.add_argument("--model_config", required=True, help="model_config.json path, comma list, glob, or auto.")
    parser.add_argument(
        "--output_dir",
        default="head_dir",
        help="Output directory. Use head_dir/same_as_head/inplace to write next to each best_head.pt.",
    )
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--clips_per_recording", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--graph_k", type=int, default=None)
    parser.add_argument("--edge_mode", choices=["temporal", "similarity", "temporal_similarity"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=["DeepShip", "ShipsEar", "auto"], default="auto")
    parser.add_argument("--loss_log", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    runs = pair_runs(args, root)
    inplace_outputs = str(args.output_dir).lower() in {"head_dir", "same_as_head", "inplace"}
    output_dir = None if inplace_outputs else resolve_path(args.output_dir, root)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for run in runs:
        temp_checkpoint = torch_load(run["graph_head_ckpt"], map_location="cpu")
        temp_config = run["model_config_payload"]
        try:
            encoder_checkpoint = run["encoder_ckpt"]
            temp_num_classes = int(torch_load(encoder_checkpoint, map_location="cpu")["cnn_cfg"]["n_class"])
        except Exception:
            temp_num_classes = 0
        dataset_name = infer_dataset_name(args.dataset, temp_config, run["encoder_ckpt"], temp_num_classes)
        if inplace_outputs:
            subdir = run["graph_head_ckpt"].parent
        else:
            subdir = output_dir
        if len(runs) > 1 and not inplace_outputs:
            subdir = output_dir / run_name(run, temp_checkpoint, dataset_name)
        print(f"Evaluating {run['graph_head_ckpt']} -> {subdir}", flush=True)
        summaries.append(evaluate_one_run(run, args, subdir, root))

    if len(summaries) > 1:
        if inplace_outputs:
            summary_dir = Path(os.path.commonpath([str(run["graph_head_ckpt"].parent) for run in runs]))
            write_multiseed_summary(summary_dir, summaries, output_prefix="review_")
        else:
            write_multiseed_summary(output_dir, summaries)
    print(json.dumps({"runs": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
