#!/usr/bin/env python3
"""Train/evaluate Orthogonal Signal-Noise Decoupling aggregation.

The ShuffleFAC encoder is loaded from an existing first-stage checkpoint and is
kept frozen. Only the signal/noise decoupler, noise graph head, attention mask,
and classifier are optimized.
"""

import argparse
import atexit
import csv
import glob
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from run_graphhead import FrozenShuffleFACEncoder, RecordingBagCachedDataset, resolve_path


DEEPSHIP_CLASS_NAMES = ["Cargo", "Passengership", "Tanker", "Tug"]
SHIPSEAR_CLASS_NAMES = ["A", "B", "C", "D", "E"]


class TeeStream:
    """Mirror writes to the original console stream and a run.log file."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"
        self.errors = getattr(streams[0], "errors", "replace") if streams else "replace"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return bool(self.streams and self.streams[0].isatty())

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def install_run_log(output_dir: Path):
    """Capture stdout/stderr in output_dir/run.log while keeping console output."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    def close_log():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()

    atexit.register(close_log)
    print(f"run log: {log_path}", flush=True)
    print(f"command: {' '.join(sys.argv)}", flush=True)


def torch_load(path: Path, map_location="cpu"):
    """Load torch checkpoints across PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict):
    """Write JSON with stable indentation and UTF-8 text."""

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def expand_encoder_ckpts(value: str, root: Path):
    """Expand comma-separated files, globs, and directories into best.pt paths."""

    paths = []
    for part in [p.strip() for p in str(value).split(",") if p.strip()]:
        if any(ch in part for ch in "*?[]"):
            paths.extend(Path(p) for p in glob.glob(part))
            continue
        path = resolve_path(part, root)
        if path.is_dir():
            paths.extend(sorted(path.rglob("best.pt")))
        else:
            paths.append(path)
    out = []
    seen = set()
    for path in sorted(paths, key=lambda p: str(p)):
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def checkpoint_args(checkpoint: dict):
    """Return checkpoint args when present, otherwise an empty dict."""

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def pick(cli_value, checkpoint_args_dict: dict, name: str, default):
    """Prefer CLI values, then checkpoint args, then a hard default."""

    if cli_value is not None:
        return cli_value
    if isinstance(checkpoint_args_dict, dict) and checkpoint_args_dict.get(name) is not None:
        return checkpoint_args_dict[name]
    return default


def cli_arg_was_passed(name: str) -> bool:
    """Return whether a CLI option was explicitly provided."""

    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def infer_model_config(encoder_ckpt: Path, root: Path, model_config_arg: str) -> Path:
    """Locate the first-stage model_config.json that stores recording cache paths."""

    if str(model_config_arg).lower() != "auto":
        return resolve_path(model_config_arg, root)
    candidate = encoder_ckpt.parent / "model_config.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not infer model_config next to encoder checkpoint: {encoder_ckpt}")
    return candidate


def resolve_cache_paths(config: dict, root: Path) -> dict:
    """Resolve train/val/test cache paths from model_config.json."""

    cache_paths = config.get("cache_paths")
    if not isinstance(cache_paths, dict):
        raise ValueError("model_config must contain cache_paths")
    for split in ["train", "val", "test"]:
        if split not in cache_paths:
            raise ValueError(f"cache_paths missing split: {split}")
    return {split: resolve_path(path, root) for split, path in cache_paths.items()}


def validate_recording_cache(cache_path: Path, split_name: str):
    """Reject frame-level or split-mismatched caches before training/evaluation."""

    payload = torch_load(cache_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    protocol = metadata.get("protocol")
    actual_split = payload.get("split_name")
    del payload
    if protocol != "recording_level":
        raise ValueError(f"{cache_path} is not recording-level cache. metadata.protocol={protocol!r}")
    if actual_split and actual_split != split_name:
        raise ValueError(f"{cache_path} split mismatch: expected {split_name}, got {actual_split}")


def infer_dataset_name(dataset_arg: str, config: dict, encoder_ckpt: Path, num_classes: int):
    """Infer DeepShip or ShipsEar from config metadata, path text, or class count."""

    if dataset_arg != "auto":
        return dataset_arg
    split_meta = config.get("split_metadata", {})
    parent = str(split_meta.get("parent_folder", ""))
    text = " ".join([str(encoder_ckpt), parent, json.dumps(split_meta.get("class_mapping", {}))])
    if "ShipsEar" in text or '"E": 4' in text:
        return "ShipsEar"
    if "DeepShip" in text or "Cargo" in text:
        return "DeepShip"
    if num_classes == 4:
        return "DeepShip"
    if num_classes == 5:
        return "ShipsEar"
    return "auto"


def class_names_for(dataset_name: str, num_classes: int):
    """Return dataset-specific class names or generic class_i names."""

    if dataset_name == "DeepShip" and num_classes == 4:
        return DEEPSHIP_CLASS_NAMES
    if dataset_name == "ShipsEar" and num_classes == 5:
        return SHIPSEAR_CLASS_NAMES
    return [f"class_{idx}" for idx in range(num_classes)]


class SignalNoiseDecoupler(nn.Module):
    """Projects frozen encoder embeddings into signal and noise subspaces."""

    def __init__(self, in_dim: int, sig_dim: int, noise_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.signal_proj = nn.Sequential(
            nn.Linear(in_dim, sig_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sig_dim, sig_dim),
            nn.LayerNorm(sig_dim),
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(in_dim, noise_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(noise_dim, noise_dim),
            nn.LayerNorm(noise_dim),
        )

    def forward(self, z):
        z = self.input_norm(z)
        return self.signal_proj(z), self.noise_proj(z)


class NoiseGraphConv(nn.Module):
    """Graph smoothing that operates only on the noise subspace."""

    def __init__(self, noise_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(noise_dim * 2, noise_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(noise_dim, noise_dim),
        )
        self.norm = nn.LayerNorm(noise_dim)

    def forward(self, z_noise, knn_idx):
        b, s, d = z_noise.shape
        k = knn_idx.size(-1)
        expanded = z_noise.unsqueeze(1).expand(b, s, s, d)
        gather_idx = knn_idx.unsqueeze(-1).expand(b, s, k, d)
        neighbors = torch.gather(expanded, dim=2, index=gather_idx)
        delta = (neighbors - z_noise.unsqueeze(2)).mean(dim=2)
        update = self.msg_mlp(torch.cat([z_noise, delta], dim=-1))
        return self.norm(update)


class NoiseGraphAttentionHead(nn.Module):
    """Builds a graph on z_noise and extracts a global bottom-noise vector."""

    def __init__(
        self,
        noise_dim: int,
        graph_k: int = 2,
        edge_mode: str = "temporal_similarity",
        sim_threshold: float = 0.8,
        use_temperature: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_k = int(graph_k)
        self.edge_mode = edge_mode
        self.sim_threshold = float(sim_threshold)
        self.graph_conv = NoiseGraphConv(noise_dim=noise_dim, dropout=dropout)
        self.graph_res_scale = nn.Parameter(torch.tensor(0.1))
        self.noise_attn = nn.Sequential(
            nn.Linear(noise_dim, max(noise_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(noise_dim // 2, 1), 1),
        )
        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_attn_entropy = torch.tensor(0.0)
        self.last_graph_delta_norm = torch.tensor(0.0)
        self.last_avg_graph_degree = torch.tensor(0.0)

    def _build_graph(self, z_noise):
        b, s, _ = z_noise.shape
        if s <= 1:
            return torch.zeros((b, s, 1), dtype=torch.long, device=z_noise.device)

        adj = torch.zeros((b, s, s), dtype=torch.bool, device=z_noise.device)
        if self.edge_mode in ("temporal", "temporal_similarity", "threshold_similarity"):
            for i in range(s):
                if i > 0:
                    adj[:, i, i - 1] = True
                if i + 1 < s:
                    adj[:, i, i + 1] = True
        if self.edge_mode == "threshold_similarity":
            normed = F.normalize(z_noise, p=2, dim=-1)
            sim = torch.bmm(normed, normed.transpose(1, 2))
            eye = torch.eye(s, dtype=torch.bool, device=z_noise.device).unsqueeze(0)
            sim = sim.masked_fill(eye, -float("inf"))
            adj = adj | (sim > self.sim_threshold)
        elif self.edge_mode in ("similarity", "temporal_similarity"):
            k = min(self.graph_k, s - 1)
            normed = F.normalize(z_noise, p=2, dim=-1)
            sim = torch.bmm(normed, normed.transpose(1, 2))
            eye = torch.eye(s, dtype=torch.bool, device=z_noise.device).unsqueeze(0)
            sim = sim.masked_fill(eye, -float("inf"))
            sim_idx = sim.topk(k=k, dim=-1).indices
            adj.scatter_(2, sim_idx, True)

        self.last_avg_graph_degree = adj.sum(dim=-1).float().mean().detach()
        max_degree = max(int(adj.sum(dim=-1).max().item()), 1)
        out = torch.zeros((b, s, max_degree), dtype=torch.long, device=z_noise.device)
        for bi in range(b):
            for i in range(s):
                idx = torch.nonzero(adj[bi, i], as_tuple=False).flatten()
                if idx.numel() == 0:
                    idx = torch.tensor([i], dtype=torch.long, device=z_noise.device)
                if idx.numel() < max_degree:
                    idx = torch.cat([idx, idx[:1].expand(max_degree - idx.numel())], dim=0)
                out[bi, i] = idx[:max_degree]
        return out

    def forward(self, z_noise):
        if z_noise.size(1) <= 1:
            smoothed = z_noise
            self.last_avg_graph_degree = z_noise.new_tensor(0.0).detach()
        else:
            knn_idx = self._build_graph(z_noise)
            update = self.graph_conv(z_noise, knn_idx)
            smoothed = z_noise + self.graph_res_scale * update
        self.last_graph_delta_norm = (smoothed - z_noise).norm(dim=-1).mean().detach()

        scores = self.noise_attn(smoothed)
        if hasattr(self, "temperature"):
            clamped_temp = torch.clamp(self.temperature, min=0.1)
            weights = torch.softmax(scores / clamped_temp, dim=1)
        else:
            weights = torch.softmax(scores, dim=1)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()
        global_noise = (weights * smoothed).sum(dim=1)
        return global_noise, smoothed, weights


class SignalNoiseDecoupledModel(nn.Module):
    """Frozen ShuffleFAC encoder plus trainable signal/noise decoupled head."""

    def __init__(
        self,
        encoder: FrozenShuffleFACEncoder,
        num_classes: int,
        sig_dim: int = 32,
        noise_dim: int = 32,
        graph_k: int = 2,
        edge_mode: str = "temporal_similarity",
        sim_threshold: float = 0.8,
        use_temperature: bool = False,
        signal_top_k: int = 0,
        topk_warmup_epochs: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_temperature = bool(use_temperature)
        self.sim_threshold = float(sim_threshold)
        self.signal_top_k = int(signal_top_k)
        self.topk_warmup_epochs = int(topk_warmup_epochs)
        self.current_epoch = 0
        self.decoupler = SignalNoiseDecoupler(encoder.embed_dim, sig_dim, noise_dim, dropout=dropout)
        self.noise_graph = NoiseGraphAttentionHead(
            noise_dim=noise_dim,
            graph_k=graph_k,
            edge_mode=edge_mode,
            sim_threshold=sim_threshold,
            use_temperature=use_temperature,
            dropout=dropout,
        )
        self.noise_to_suppression = nn.Sequential(
            nn.LayerNorm(noise_dim),
            nn.Linear(noise_dim, sig_dim),
            nn.Sigmoid(),
        )
        self.signal_attn = nn.Sequential(
            nn.Linear(sig_dim, max(sig_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(sig_dim // 2, 1), 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(sig_dim, num_classes))
        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(1.0))
        self.last_attn_entropy = torch.tensor(0.0)
        self.last_graph_delta_norm = torch.tensor(0.0)
        self.last_avg_graph_degree = torch.tensor(0.0)
        self.last_signal_topk_count = torch.tensor(0.0)
        self.graph_res_scale = self.noise_graph.graph_res_scale

        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode_nodes(self, clips):
        """Encode clip bags with the frozen ShuffleFAC CNN under no_grad."""

        b, s, c, f, t = clips.shape
        flat = clips.reshape(b * s, c, f, t)
        with torch.no_grad():
            emb = self.encoder(flat).view(b, s, -1)
        return emb.detach()

    def topk_masked_softmax(self, signal_scores):
        """Apply optional Top-K sparse signal evidence pooling before softmax."""

        s = signal_scores.size(1)
        apply_topk = False
        if hasattr(self, "signal_top_k") and self.signal_top_k > 0 and s > 1:
            apply_topk = True
            if (
                self.training
                and hasattr(self, "current_epoch")
                and getattr(self, "current_epoch", 1) <= getattr(self, "topk_warmup_epochs", 0)
            ):
                apply_topk = False

        if apply_topk:
            k = min(self.signal_top_k, s)
            scores_2d = signal_scores.squeeze(-1)
            topk_idx = scores_2d.topk(k=k, dim=1).indices
            mask_2d = torch.zeros_like(scores_2d, dtype=torch.bool)
            mask_2d.scatter_(1, topk_idx, True)
            min_val = torch.finfo(signal_scores.dtype).min
            masked_scores = signal_scores.masked_fill(~mask_2d.unsqueeze(-1), min_val)
            signal_topk_mask = mask_2d
        else:
            masked_scores = signal_scores
            signal_topk_mask = torch.ones_like(signal_scores.squeeze(-1), dtype=torch.bool)

        if hasattr(self, "temperature"):
            clamped_temp = torch.clamp(self.temperature, min=0.1)
            signal_weights = torch.softmax(masked_scores / clamped_temp, dim=1)
        else:
            signal_weights = torch.softmax(masked_scores, dim=1)

        return signal_weights, signal_topk_mask

    def forward(self, clips, return_parts: bool = False):
        """Classify a recording bag after noise-conditioned signal masking."""

        z = self.encode_nodes(clips)
        z_sig, z_noise = self.decoupler(z)

        global_noise, z_noise_smooth, noise_weights = self.noise_graph(z_noise)
        suppression = self.noise_to_suppression(global_noise).unsqueeze(1)
        z_sig_filtered = z_sig * (1.0 - suppression)

        signal_scores = self.signal_attn(z_sig_filtered)
        signal_weights, signal_topk_mask = self.topk_masked_softmax(signal_scores)
        entropy = -(signal_weights * (signal_weights + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()
        self.last_graph_delta_norm = self.noise_graph.last_graph_delta_norm.detach()
        self.last_avg_graph_degree = self.noise_graph.last_avg_graph_degree.detach()
        self.last_signal_topk_count = signal_topk_mask.float().sum(dim=1).mean().detach()

        recording_signal = (signal_weights * z_sig_filtered).sum(dim=1)
        logits = self.classifier(recording_signal)

        if not return_parts:
            return logits
        return logits, {
            "z_sig": z_sig,
            "z_noise": z_noise,
            "z_noise_smooth": z_noise_smooth,
            "global_noise": global_noise,
            "suppression": suppression.squeeze(1),
            "signal_weights": signal_weights,
            "signal_topk_mask": signal_topk_mask,
            "noise_weights": noise_weights,
        }


def orthogonal_loss(z_sig, z_noise):
    """Penalize dot products between signal and noise subspace projections."""

    dim = min(z_sig.size(-1), z_noise.size(-1))
    if dim <= 0:
        return z_sig.new_tensor(0.0)
    sig = F.normalize(z_sig[..., :dim], p=2, dim=-1)
    noise = F.normalize(z_noise[..., :dim], p=2, dim=-1)
    dot = (sig * noise).sum(dim=-1)
    return (dot.pow(2)).mean()


def noise_consistency_loss(z_noise, slice_ids=None):
    """Force same-recording noise projections to approach their bag mean."""

    del slice_ids
    mean_noise = z_noise.mean(dim=1, keepdim=True)
    return F.mse_loss(z_noise, mean_noise.expand_as(z_noise))


def compute_losses(logits, labels, parts, criterion, lambda_orth: float, lambda_noise_consistency: float):
    """Compute task, orthogonal, noise-consistency, and weighted total losses."""

    task = criterion(logits, labels)
    orth = orthogonal_loss(parts["z_sig"], parts["z_noise"])
    noise_consistency = noise_consistency_loss(parts["z_noise"])
    total = task + float(lambda_orth) * orth + float(lambda_noise_consistency) * noise_consistency
    return {
        "total": total,
        "task": task,
        "orth": orth,
        "noise_consistency": noise_consistency,
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable numpy softmax for saved probabilities."""

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def metrics_from_arrays(y_true, y_pred):
    """Compute recording-level classification metrics from labels and predictions."""

    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Precision macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Precision weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    lambda_orth,
    lambda_noise_consistency,
    grad_clip: float = 5.0,
):
    """Train one downstream epoch; skip non-finite loss batches safely."""

    model.train()
    model.encoder.encoder.eval()
    totals = {"total": 0.0, "task": 0.0, "orth": 0.0, "noise_consistency": 0.0}
    y_true = []
    logits_all = []
    n = 0
    skipped_nan = 0
    entropy_vals = []
    delta_vals = []
    degree_vals = []
    topk_count_vals = []
    for clips, labels, _rids in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, parts = model(clips, return_parts=True)
        losses = compute_losses(logits, labels, parts, criterion, lambda_orth, lambda_noise_consistency)
        if not torch.isfinite(losses["total"]).all():
            skipped_nan += 1
            print("Warning: NaN/Inf detected in loss; skipping batch.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        losses["total"].backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(grad_clip),
            )
        optimizer.step()

        batch = int(labels.size(0))
        n += batch
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * batch
        y_true.extend(labels.detach().cpu().numpy().tolist())
        logits_all.append(logits.detach().cpu())
        entropy_vals.append(float(model.last_attn_entropy.detach().cpu()))
        delta_vals.append(float(model.last_graph_delta_norm.detach().cpu()))
        degree_vals.append(float(model.last_avg_graph_degree.detach().cpu()))
        topk_count_vals.append(float(model.last_signal_topk_count.detach().cpu()))

    if not logits_all:
        out = {
            "ACC": math.nan,
            "Macro-F1": math.nan,
            "Weighted-F1": math.nan,
            "Precision macro": math.nan,
            "Precision weighted": math.nan,
            "Recall macro": math.nan,
            "Recall weighted": math.nan,
            "total_loss": math.nan,
            "task_loss": math.nan,
            "orth_loss": math.nan,
            "noise_consistency_loss": math.nan,
            "attn_entropy": math.nan,
            "graph_delta_norm": math.nan,
            "avg_graph_degree": math.nan,
            "avg_signal_topk_count": math.nan,
            "graph_res_scale": float(model.graph_res_scale.detach().cpu()),
            "skipped_nan_batches": skipped_nan,
        }
        if hasattr(model, "temperature"):
            out["learned_temperature"] = float(model.temperature.detach().cpu())
        return out

    logits_np = torch.cat(logits_all, dim=0).numpy()
    pred_np = logits_np.argmax(axis=1)
    out = metrics_from_arrays(np.asarray(y_true), pred_np)
    for key in totals:
        out[f"{key}_loss"] = totals[key] / max(n, 1)
    out["attn_entropy"] = float(np.mean(entropy_vals)) if entropy_vals else math.nan
    out["graph_delta_norm"] = float(np.mean(delta_vals)) if delta_vals else math.nan
    out["avg_graph_degree"] = float(np.mean(degree_vals)) if degree_vals else math.nan
    out["avg_signal_topk_count"] = float(np.mean(topk_count_vals)) if topk_count_vals else math.nan
    out["graph_res_scale"] = float(model.graph_res_scale.detach().cpu())
    if hasattr(model, "temperature"):
        out["learned_temperature"] = float(model.temperature.detach().cpu())
    out["skipped_nan_batches"] = skipped_nan
    return out


@torch.no_grad()
def collect_multisample_predictions(
    model,
    dataset,
    batch_size,
    criterion,
    device,
    eval_samples: int,
    lambda_orth: float,
    lambda_noise_consistency: float,
):
    """Evaluate recording-level predictions with deterministic multi-bag voting."""

    model.eval()
    model.current_epoch = 99999
    model.encoder.encoder.eval()
    y_true = []
    logits_all = []
    recording_ids = []
    totals = {"total": 0.0, "task": 0.0, "orth": 0.0, "noise_consistency": 0.0}
    entropy_vals = []
    delta_vals = []
    degree_vals = []
    topk_count_vals = []
    n = 0

    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        labels = torch.tensor([int(dataset.recordings[i][2]) for i in indices], dtype=torch.long, device=device)
        rids = [str(dataset.recordings[i][0]) for i in indices]

        sample_logits = []
        sample_orth = []
        sample_noise = []
        sample_entropy = []
        sample_delta = []
        sample_degree = []
        sample_topk_count = []
        for sample_id in range(eval_samples):
            clips = torch.stack(
                [dataset.get_eval_item(i, sample_id, eval_samples)[0] for i in indices],
                dim=0,
            ).to(device, non_blocking=True)
            logits, parts = model(clips, return_parts=True)
            sample_logits.append(logits.detach())
            sample_orth.append(orthogonal_loss(parts["z_sig"], parts["z_noise"]).detach())
            sample_noise.append(noise_consistency_loss(parts["z_noise"]).detach())
            sample_entropy.append(float(model.last_attn_entropy.detach().cpu()))
            sample_delta.append(float(model.last_graph_delta_norm.detach().cpu()))
            sample_degree.append(float(model.last_avg_graph_degree.detach().cpu()))
            sample_topk_count.append(float(model.last_signal_topk_count.detach().cpu()))

        logits = torch.stack(sample_logits, dim=0).mean(dim=0)
        task = criterion(logits, labels)
        orth = torch.stack(sample_orth).mean()
        noise = torch.stack(sample_noise).mean()
        total = task + float(lambda_orth) * orth + float(lambda_noise_consistency) * noise

        batch = int(labels.size(0))
        n += batch
        totals["total"] += float(total.detach().cpu()) * batch
        totals["task"] += float(task.detach().cpu()) * batch
        totals["orth"] += float(orth.detach().cpu()) * batch
        totals["noise_consistency"] += float(noise.detach().cpu()) * batch
        y_true.extend(labels.detach().cpu().numpy().tolist())
        logits_all.append(logits.detach().cpu())
        recording_ids.extend(rids)
        entropy_vals.append(float(np.mean(sample_entropy)))
        delta_vals.append(float(np.mean(sample_delta)))
        degree_vals.append(float(np.mean(sample_degree)))
        topk_count_vals.append(float(np.mean(sample_topk_count)))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    logits_np = torch.cat(logits_all, dim=0).numpy()
    prob_np = softmax_np(logits_np)
    pred_np = logits_np.argmax(axis=1).astype(np.int64)
    out = {
        "y_true": y_true_np,
        "y_pred": pred_np,
        "y_logits": logits_np,
        "y_prob": prob_np,
        "recording_ids": recording_ids,
        "attn_entropy": float(np.mean(entropy_vals)) if entropy_vals else math.nan,
        "graph_delta_norm": float(np.mean(delta_vals)) if delta_vals else math.nan,
        "avg_graph_degree": float(np.mean(degree_vals)) if degree_vals else math.nan,
        "avg_signal_topk_count": float(np.mean(topk_count_vals)) if topk_count_vals else math.nan,
        "graph_res_scale": float(model.graph_res_scale.detach().cpu()),
    }
    if hasattr(model, "temperature"):
        out["learned_temperature"] = float(model.temperature.detach().cpu())
    out.update(metrics_from_arrays(y_true_np, pred_np))
    for key in totals:
        out[f"{key}_loss"] = totals[key] / max(n, 1)
    return out


def write_matrix_csv(path: Path, matrix: np.ndarray, class_names):
    """Save a labeled confusion matrix CSV."""

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for idx, row in enumerate(matrix):
            writer.writerow([class_names[idx], *row.tolist()])


def plot_confusion(path: Path, matrix: np.ndarray, class_names, normalized: bool):
    """Render a publication-readable confusion matrix using matplotlib only."""

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.4), max(5, len(class_names) * 1.2)))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(
        "Signal-Noise Decoupled Recording-level Confusion Matrix"
        + (" (Row-normalized)" if normalized else " (Counts)")
    )
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
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_confusion_outputs(out_dir: Path, y_true, y_pred, class_names):
    """Save raw-count and row-normalized confusion matrices as PNG and CSV."""

    labels = list(range(len(class_names)))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = counts.astype(np.float64) / counts.sum(axis=1, keepdims=True)
    normalized = np.nan_to_num(normalized)
    write_matrix_csv(out_dir / "confusion_matrix_counts.csv", counts, class_names)
    write_matrix_csv(out_dir / "confusion_matrix_normalized.csv", normalized, class_names)
    plot_confusion(out_dir / "confusion_matrix_counts.png", counts, class_names, normalized=False)
    plot_confusion(out_dir / "confusion_matrix_normalized.png", normalized, class_names, normalized=True)


def save_predictions(out_dir: Path, metrics: dict, class_names):
    """Save recording-level labels, logits, probabilities, and report files."""

    np.save(out_dir / "y_true.npy", metrics["y_true"])
    np.save(out_dir / "y_pred.npy", metrics["y_pred"])
    np.save(out_dir / "y_logits.npy", metrics["y_logits"])
    np.save(out_dir / "y_prob.npy", metrics["y_prob"])

    fields = ["recording_id", "true_label", "pred_label", "true_name", "pred_name", "confidence"]
    fields += [f"logit_class_{idx}" for idx in range(len(class_names))]
    fields += [f"prob_class_{idx}" for idx in range(len(class_names))]
    with (out_dir / "predictions_recording_level.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")


def save_loss_curves(out_dir: Path, rows, encoder_ckpt: Path):
    """Save downstream loss/F1 curves or a clear missing-curve note."""

    if rows:
        with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        epoch = np.asarray([float(row["epoch"]) for row in rows], dtype=np.float64)
        train_total = np.asarray([float(row["train_total_loss"]) for row in rows], dtype=np.float64)
        val_total = np.asarray([float(row["val_total_loss"]) for row in rows], dtype=np.float64)
        val_macro = np.asarray([float(row["val_macro_f1"]) for row in rows], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epoch, train_total, label="train_total_loss", linewidth=2)
        ax.plot(epoch, val_total, label="val_total_loss", linewidth=2)
        ax.set_title("Signal-Noise Decoupled Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epoch, val_macro, label="val_macro_f1", linewidth=2)
        ax.set_title("Signal-Noise Decoupled Val Macro-F1 Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Macro-F1")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "val_macro_f1_curve.png", dpi=220)
        plt.close(fig)
    else:
        (out_dir / "loss_curve_missing.txt").write_text(
            "loss curve source not found. No training epochs were run and no epoch_metrics.csv was generated.\n",
            encoding="utf-8",
        )

    train_log = encoder_ckpt.parent / "train_log.txt"
    if train_log.exists():
        try:
            with train_log.open("r", encoding="utf-8", newline="") as f:
                enc_rows = list(csv.DictReader(f))
            if enc_rows:
                epoch = np.asarray([float(row["epoch"]) for row in enc_rows], dtype=np.float64)
                train_loss = np.asarray([float(row["train_loss"]) for row in enc_rows], dtype=np.float64)
                val_loss = np.asarray([float(row["val_loss"]) for row in enc_rows], dtype=np.float64)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(epoch, train_loss, label="encoder_train_loss", linewidth=2)
                ax.plot(epoch, val_loss, label="encoder_val_loss", linewidth=2)
                ax.set_title("Frozen ShuffleFAC Encoder Loss Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "loss_curve_encoder.png", dpi=220)
                plt.close(fig)
        except Exception as exc:
            (out_dir / "loss_curve_encoder_missing.txt").write_text(str(exc) + "\n", encoding="utf-8")


def count_params(model):
    """Count total, trainable, frozen, encoder, and downstream head parameters."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "encoder_params": int(encoder_params),
        "head_params": int(total - encoder_params),
    }


def format_big_number(value: Optional[int]):
    """Format large operation counts for metrics files."""

    if value is None:
        return None
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def count_macs_hooks(model, dummy_input):
    """Fallback Conv2d/Linear MAC counter based on forward hooks."""

    hooks = []
    macs = {"total": 0}

    def conv_hook(module, inputs, output):
        x = inputs[0]
        out = output[0] if isinstance(output, tuple) else output
        if not isinstance(out, torch.Tensor) or out.ndim < 4:
            return
        batch = int(x.shape[0])
        out_channels = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        kernel_ops = int(module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups))
        macs["total"] += batch * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
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


def count_macs_fvcore(model, dummy_input):
    """Try fvcore FlopCountAnalysis and treat reported FLOPs as MAC-like ops."""

    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        raise RuntimeError(f"fvcore unavailable: {exc}") from exc
    analysis = FlopCountAnalysis(model, dummy_input)
    return int(analysis.total())


def compute_complexity(model, dataset, device):
    """Compute MACs from a real cached sample, preferring fvcore when available."""

    try:
        clips, _label, _rid = dataset.get_eval_item(0, 0, 1)
        dummy = clips.unsqueeze(0).to(device)
        try:
            macs = count_macs_fvcore(model, dummy)
            method = "fvcore_flop_count_analysis"
        except Exception as fvcore_exc:
            macs = count_macs_hooks(model, dummy)
            method = f"conv_linear_forward_hooks (fvcore fallback: {fvcore_exc})"
        return {
            "macs_available": True,
            "input_shape": list(dummy.shape),
            "macs": int(macs),
            "macs_formatted": format_big_number(int(macs)),
            "macs_method": method,
        }
    except Exception as exc:
        return {
            "macs_available": False,
            "reason": str(exc),
            "macs": None,
            "macs_formatted": None,
            "macs_method": None,
        }


def save_param_outputs(out_dir: Path, param_summary: dict, complexity: dict):
    """Save parameter and complexity summaries in txt/json/csv formats."""

    write_json(out_dir / "params_summary.json", param_summary)
    write_json(out_dir / "complexity_summary.json", complexity)
    with (out_dir / "params_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(param_summary.keys()))
        writer.writeheader()
        writer.writerow(param_summary)
    with (out_dir / "params_summary.txt").open("w", encoding="utf-8") as f:
        for key, value in param_summary.items():
            f.write(f"{key}: {value}\n")
        f.write(f"macs_available: {complexity.get('macs_available')}\n")
        f.write(f"macs: {complexity.get('macs')}\n")
        f.write(f"macs_formatted: {complexity.get('macs_formatted')}\n")
        f.write(f"macs_method: {complexity.get('macs_method')}\n")
        if not complexity.get("macs_available"):
            f.write(f"macs_reason: {complexity.get('reason')}\n")


def write_metrics_txt(path: Path, metrics: dict):
    """Save a flat metrics dictionary as human-readable text."""

    with path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
            else:
                f.write(f"{key}: {value}\n")


def save_split_outputs(out_dir: Path, pred_metrics: dict, class_names):
    """Save all per-split prediction and confusion-matrix artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_outputs(out_dir, pred_metrics["y_true"], pred_metrics["y_pred"], class_names)
    save_predictions(out_dir, pred_metrics, class_names)


def load_head_checkpoint(model, ckpt_path: Path, out_dir: Path, device):
    """Load a downstream checkpoint and record strict-load warnings."""

    checkpoint = torch_load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state is None:
        state = checkpoint.get("state_dict")
    if state is None:
        state = checkpoint
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    warning_path = out_dir / "load_warnings.txt"
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
    return checkpoint


def seed_from_path_or_checkpoint(path: Path, checkpoint: dict):
    """Extract a stable seed identifier from checkpoint args or path text."""

    args = checkpoint_args(checkpoint)
    if "training_seed" in args:
        return str(args["training_seed"])
    if "random_seed" in args:
        return str(args["random_seed"])
    if "seed" in args:
        return str(args["seed"])
    match = re.search(r"seed[_-]?(\d+)", str(path))
    return match.group(1) if match else "unknown"


def run_one(encoder_ckpt: Path, args, output_dir: Path, root: Path):
    """Run one seed in either train-then-evaluate mode or eval-only mode."""

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_payload = torch_load(encoder_ckpt, map_location="cpu")
    model_config_path = infer_model_config(encoder_ckpt, root, args.model_config)
    config = load_json(model_config_path)
    head_payload = torch_load(resolve_path(args.head_ckpt, root), map_location="cpu") if args.head_ckpt else {}
    head_args = checkpoint_args(head_payload)

    batch_size = int(pick(args.batch_size, head_args, "batch_size", 16))
    eval_samples = int(pick(args.eval_samples, head_args, "eval_samples", 5))
    clips_per_recording = int(pick(args.clips_per_recording, head_args, "clips_per_recording", 8))
    seed = int(pick(args.seed, head_args, "seed", checkpoint_args(encoder_payload).get("training_seed", 42)))
    graph_k = int(pick(args.graph_k, head_args, "graph_k", 2))
    edge_mode = str(pick(args.edge_mode, head_args, "edge_mode", "temporal_similarity"))
    if not cli_arg_was_passed("--sim_threshold") and isinstance(head_args, dict) and head_args.get("sim_threshold") is not None:
        sim_threshold = float(head_args["sim_threshold"])
    else:
        sim_threshold = float(args.sim_threshold)
    use_temperature = bool(pick(args.use_temperature, head_args, "use_temperature", False))
    if not cli_arg_was_passed("--signal_top_k") and isinstance(head_args, dict) and head_args.get("signal_top_k") is not None:
        signal_top_k = int(head_args["signal_top_k"])
    else:
        signal_top_k = int(args.signal_top_k)
    if (
        not cli_arg_was_passed("--topk_warmup_epochs")
        and isinstance(head_args, dict)
        and head_args.get("topk_warmup_epochs") is not None
    ):
        topk_warmup_epochs = int(head_args["topk_warmup_epochs"])
    else:
        topk_warmup_epochs = int(args.topk_warmup_epochs)
    dropout = float(pick(args.dropout, head_args, "dropout", 0.1))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    cache_paths = resolve_cache_paths(config, root)
    for split_name in ["train", "val", "test"]:
        validate_recording_cache(cache_paths[split_name], split_name)

    train_set = RecordingBagCachedDataset(cache_paths["train"], clips_per_recording, train=True, seed=seed)
    val_set = RecordingBagCachedDataset(cache_paths["val"], clips_per_recording, train=False, seed=seed)
    test_set = RecordingBagCachedDataset(cache_paths["test"], clips_per_recording, train=False, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    encoder = FrozenShuffleFACEncoder(encoder_ckpt, device=device)
    for param in encoder.parameters():
        param.requires_grad = False
    num_classes = int(encoder.cnn_cfg["n_class"])
    dataset_name = infer_dataset_name(args.dataset, config, encoder_ckpt, num_classes)
    class_names = class_names_for(dataset_name, num_classes)
    model = SignalNoiseDecoupledModel(
        encoder=encoder,
        num_classes=num_classes,
        sig_dim=args.sig_dim,
        noise_dim=args.noise_dim,
        graph_k=graph_k,
        edge_mode=edge_mode,
        sim_threshold=sim_threshold,
        use_temperature=use_temperature,
        signal_top_k=signal_top_k,
        topk_warmup_epochs=topk_warmup_epochs,
        dropout=dropout,
    ).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    best_path = output_dir / "best_signal_noise_decoupled.pt"
    epoch_rows = []
    best_val = -1.0
    best_epoch = -1
    stale = 0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    mode = "eval" if args.eval_only else args.mode

    if mode == "eval":
        if not args.head_ckpt:
            raise ValueError("--mode eval requires --head_ckpt")
        load_head_checkpoint(model, resolve_path(args.head_ckpt, root), output_dir, device)
        best_path = resolve_path(args.head_ckpt, root)
        if isinstance(head_payload, dict):
            best_val = float(head_payload.get("best_val_macro_f1", best_val))
            best_epoch = int(head_payload.get("epoch", best_epoch))
        print(f"Mode: eval. Loaded downstream checkpoint: {args.head_ckpt}", flush=True)
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if args.head_ckpt:
            load_head_checkpoint(model, resolve_path(args.head_ckpt, root), output_dir, device)
            print(f"Mode: train. Warm-started downstream checkpoint: {args.head_ckpt}", flush=True)
        print(f"Loaded frozen encoder: {encoder_ckpt}", flush=True)
        print(f"Model config: {model_config_path}", flush=True)
        print(f"Recordings train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}", flush=True)
        print(f"Downstream trainable params: {trainable_params}", flush=True)
        print(f"Frozen encoder params: {frozen_params}", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.current_epoch = epoch
            train_metrics = run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.lambda_orth,
                args.lambda_noise_consistency,
                grad_clip=args.grad_clip,
            )
            val_metrics = collect_multisample_predictions(
                model,
                val_set,
                batch_size,
                criterion,
                device,
                eval_samples,
                args.lambda_orth,
                args.lambda_noise_consistency,
            )
            row = {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "train_task_loss": train_metrics["task_loss"],
                "train_orth_loss": train_metrics["orth_loss"],
                "train_noise_consistency_loss": train_metrics["noise_consistency_loss"],
                "train_acc": train_metrics["ACC"],
                "train_macro_f1": train_metrics["Macro-F1"],
                "train_skipped_nan_batches": train_metrics.get("skipped_nan_batches", 0),
                "train_avg_signal_topk_count": train_metrics["avg_signal_topk_count"],
                "val_total_loss": val_metrics["total_loss"],
                "val_task_loss": val_metrics["task_loss"],
                "val_orth_loss": val_metrics["orth_loss"],
                "val_noise_consistency_loss": val_metrics["noise_consistency_loss"],
                "val_acc": val_metrics["ACC"],
                "val_macro_f1": val_metrics["Macro-F1"],
                "val_weighted_f1": val_metrics["Weighted-F1"],
                "attn_entropy": val_metrics["attn_entropy"],
                "graph_delta_norm": val_metrics["graph_delta_norm"],
                "avg_graph_degree": val_metrics["avg_graph_degree"],
                "avg_signal_topk_count": val_metrics["avg_signal_topk_count"],
                "graph_res_scale": val_metrics["graph_res_scale"],
            }
            if "learned_temperature" in train_metrics:
                row["train_learned_temperature"] = train_metrics["learned_temperature"]
            if "learned_temperature" in val_metrics:
                row["val_learned_temperature"] = val_metrics["learned_temperature"]
            epoch_rows.append(row)
            print(
                f"{epoch},{row['train_total_loss']:.6f},{row['val_total_loss']:.6f},"
                f"{row['val_acc']:.6f},{row['val_macro_f1']:.6f},"
                f"orth={row['val_orth_loss']:.6f},noise={row['val_noise_consistency_loss']:.6f},"
                f"attn={row['attn_entropy']:.6f},delta={row['graph_delta_norm']:.6f},"
                f"degree={row['avg_graph_degree']:.6f},sig_topk={row['avg_signal_topk_count']:.6f}",
                flush=True,
            )
            if math.isfinite(val_metrics["Macro-F1"]) and val_metrics["Macro-F1"] > best_val:
                best_val = val_metrics["Macro-F1"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_val_macro_f1": best_val,
                        "args": {
                            **vars(args),
                            "batch_size": batch_size,
                            "eval_samples": eval_samples,
                            "clips_per_recording": clips_per_recording,
                            "seed": seed,
                            "graph_k": graph_k,
                            "edge_mode": edge_mode,
                            "sim_threshold": sim_threshold,
                            "use_temperature": use_temperature,
                            "signal_top_k": signal_top_k,
                            "topk_warmup_epochs": topk_warmup_epochs,
                            "dropout": dropout,
                        },
                        "encoder_ckpt": str(encoder_ckpt),
                        "model_config": str(model_config_path),
                        "trainable_params": trainable_params,
                        "frozen_params": frozen_params,
                    },
                    best_path,
                )
            else:
                stale += 1
                if args.patience > 0 and stale >= args.patience:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                    break
        if not best_path.exists():
            torch.save(
                {
                    "epoch": 0,
                    "model_state": model.state_dict(),
                    "best_val_macro_f1": best_val,
                    "args": {
                        **vars(args),
                        "batch_size": batch_size,
                        "eval_samples": eval_samples,
                        "clips_per_recording": clips_per_recording,
                        "seed": seed,
                        "graph_k": graph_k,
                        "edge_mode": edge_mode,
                        "sim_threshold": sim_threshold,
                        "use_temperature": use_temperature,
                        "signal_top_k": signal_top_k,
                        "topk_warmup_epochs": topk_warmup_epochs,
                        "dropout": dropout,
                    },
                    "encoder_ckpt": str(encoder_ckpt),
                    "model_config": str(model_config_path),
                },
                best_path,
            )
        load_head_checkpoint(model, best_path, output_dir, device)

    save_loss_curves(output_dir, epoch_rows, encoder_ckpt)

    param_summary = count_params(model)
    first_eval_set = test_set if args.split in ("test", "both") else val_set
    complexity = compute_complexity(model, first_eval_set, device)
    save_param_outputs(output_dir, param_summary, complexity)

    split_names = ["val", "test"] if args.split == "both" else [args.split]
    final_payloads = {}
    for split_name in split_names:
        dataset = val_set if split_name == "val" else test_set
        split_out = output_dir if len(split_names) == 1 else output_dir / split_name
        pred_metrics = collect_multisample_predictions(
            model,
            dataset,
            batch_size,
            criterion,
            device,
            eval_samples,
            args.lambda_orth,
            args.lambda_noise_consistency,
        )
        save_split_outputs(split_out, pred_metrics, class_names)
        prefix = "test" if split_name == "test" else split_name
        payload = {
            "split": split_name,
            "head_type": "signal_noise_decoupled",
            "mode": mode,
            "encoder_ckpt": str(encoder_ckpt),
            "head_ckpt": str(resolve_path(args.head_ckpt, root)) if args.head_ckpt else str(best_path),
            "model_config": str(model_config_path),
            "dataset": dataset_name,
            "sig_dim": args.sig_dim,
            "noise_dim": args.noise_dim,
            "lambda_orth": args.lambda_orth,
            "lambda_noise_consistency": args.lambda_noise_consistency,
            "clips_per_recording": clips_per_recording,
            "batch_size": batch_size,
            "eval_samples": eval_samples,
            "graph_k": graph_k,
            "edge_mode": edge_mode,
            "sim_threshold": sim_threshold,
            "use_temperature": use_temperature,
            "signal_top_k": signal_top_k,
            "topk_warmup_epochs": topk_warmup_epochs,
            "dropout": dropout,
            "grad_clip": args.grad_clip,
            f"{prefix}_loss": pred_metrics["total_loss"],
            f"{prefix}_task_loss": pred_metrics["task_loss"],
            f"{prefix}_orth_loss": pred_metrics["orth_loss"],
            f"{prefix}_noise_consistency_loss": pred_metrics["noise_consistency_loss"],
            f"{prefix}_acc": pred_metrics["ACC"],
            f"{prefix}_macro_f1": pred_metrics["Macro-F1"],
            f"{prefix}_weighted_f1": pred_metrics["Weighted-F1"],
            "precision_macro": pred_metrics["Precision macro"],
            "precision_weighted": pred_metrics["Precision weighted"],
            "recall_macro": pred_metrics["Recall macro"],
            "recall_weighted": pred_metrics["Recall weighted"],
            "attn_entropy": pred_metrics["attn_entropy"],
            "graph_delta_norm": pred_metrics["graph_delta_norm"],
            "avg_graph_degree": pred_metrics["avg_graph_degree"],
            "avg_signal_topk_count": pred_metrics["avg_signal_topk_count"],
            "graph_res_scale": pred_metrics["graph_res_scale"],
            **param_summary,
            "macs": complexity.get("macs"),
            "macs_formatted": complexity.get("macs_formatted"),
            "macs_method": complexity.get("macs_method"),
            "macs_available": complexity.get("macs_available"),
            "class_names": class_names,
            "best_val_macro_f1": best_val,
            "best_epoch": best_epoch,
        }
        if "learned_temperature" in pred_metrics:
            payload["learned_temperature"] = pred_metrics["learned_temperature"]
        if split_name != "test":
            payload["test_loss"] = None
            payload["test_acc"] = None
            payload["test_macro_f1"] = None
            payload["test_weighted_f1"] = None
        write_json(split_out / "metrics.json", payload)
        write_metrics_txt(split_out / "metrics.txt", payload)
        final_payloads[split_name] = payload

    if len(final_payloads) > 1:
        write_json(output_dir / "split_summary.json", final_payloads)

    used = [
        f"encoder_ckpt: {encoder_ckpt}",
        f"model_config: {model_config_path}",
        f"downstream_ckpt: {args.head_ckpt if args.head_ckpt else best_path}",
        f"train_cache: {cache_paths['train']}",
        f"val_cache: {cache_paths['val']}",
        f"test_cache: {cache_paths['test']}",
    ]
    (output_dir / "used_checkpoints.txt").write_text("\n".join(used) + "\n", encoding="utf-8")

    summary_split = "test" if "test" in final_payloads else next(iter(final_payloads))
    summary = final_payloads[summary_split]
    return {
        "run_name": output_dir.name,
        "seed": seed_from_path_or_checkpoint(encoder_ckpt, encoder_payload),
        "dataset": dataset_name,
        "split": summary_split,
        "encoder_ckpt": str(encoder_ckpt),
        "metric_acc": summary.get(f"{summary_split}_acc"),
        "metric_macro_f1": summary.get(f"{summary_split}_macro_f1"),
        "metric_weighted_f1": summary.get(f"{summary_split}_weighted_f1"),
        "total_params": param_summary["total_params"],
        "trainable_params": param_summary["trainable_params"],
        "macs": complexity.get("macs"),
        "macs_formatted": complexity.get("macs_formatted"),
    }


def write_multiseed_summary(output_dir: Path, summaries):
    """Save aggregate metrics for multi-seed runs."""

    if len(summaries) <= 1:
        return
    write_json(output_dir / "multiseed_summary.json", {"runs": summaries})
    with (output_dir / "multiseed_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def parse_args():
    """Parse CLI arguments for train/eval signal-noise decoupled runs."""

    parser = argparse.ArgumentParser(description="Train/evaluate Orthogonal Signal-Noise Decoupling aggregation.")
    parser.add_argument("--encoder_ckpt", required=True, help="First-stage ShuffleFAC best.pt path, glob, comma list, or directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="train: optimize downstream modules then evaluate; eval: load --head_ckpt and only evaluate.")
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--sig_dim", type=int, default=32)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--lambda_noise_consistency", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=["DeepShip", "ShipsEar", "auto"], default="auto")
    parser.add_argument("--model_config", default="auto", help="model_config.json path or auto to use encoder_ckpt parent.")
    parser.add_argument("--head_ckpt", default=None, help="Optional trained signal-noise-decoupled checkpoint for eval-only/resume evaluation.")
    parser.add_argument("--eval_only", action="store_true", help="Deprecated alias for --mode eval.")
    parser.add_argument("--clips_per_recording", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Clip downstream gradients; <=0 disables clipping.")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--graph_k", type=int, default=None)
    parser.add_argument("--sim_threshold", type=float, default=0.8, help="Cosine similarity threshold for threshold_similarity graph mode.")
    parser.add_argument(
        "--edge_mode",
        choices=["temporal", "similarity", "temporal_similarity", "threshold_similarity"],
        default=None,
    )
    parser.add_argument(
        "--use_temperature",
        action="store_true",
        default=None,
        help="Enable learnable temperature scaling for noise and signal attention softmax.",
    )
    parser.add_argument(
        "--signal_top_k",
        type=int,
        default=0,
        help="Use Top-K sparse signal evidence pooling; 0 keeps full soft attention.",
    )
    parser.add_argument(
        "--topk_warmup_epochs",
        type=int,
        default=0,
        help="Use full signal soft attention for this many initial training epochs before applying Top-K.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    """Entry point: expand seeds, run each seed, and write a summary."""

    args = parse_args()
    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_run_log(output_dir)

    encoder_paths = expand_encoder_ckpts(args.encoder_ckpt, root)
    if not encoder_paths:
        raise FileNotFoundError(f"No encoder checkpoints matched: {args.encoder_ckpt}")

    summaries = []
    for encoder_ckpt in encoder_paths:
        if not encoder_ckpt.exists():
            raise FileNotFoundError(f"encoder_ckpt not found: {encoder_ckpt}")
        if len(encoder_paths) > 1:
            checkpoint = torch_load(encoder_ckpt, map_location="cpu")
            seed = seed_from_path_or_checkpoint(encoder_ckpt, checkpoint)
            run_dir = output_dir / f"seed{seed}"
        else:
            run_dir = output_dir
        print(f"Running signal-noise decoupled aggregation: {encoder_ckpt} -> {run_dir}", flush=True)
        summaries.append(run_one(encoder_ckpt, args, run_dir, root))

    write_multiseed_summary(output_dir, summaries)
    print(json.dumps({"runs": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
