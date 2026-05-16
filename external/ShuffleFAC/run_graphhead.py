"""Frozen ShuffleFAC encoder with lightweight recording-level heads.

This script is intentionally separate from the main training entry points. It
loads an external ShuffleFAC checkpoint, freezes the CNN encoder, and trains
only a small recording-level aggregation head over bags of cached 3s clips.
"""

import argparse
import atexit
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from model.shuffleFAC import shuffleFAC


class TeeStream:
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
    return log_path


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return root / p


class RecordingBagCachedDataset(Dataset):
    """Groups cached clip features by recording_id and returns fixed-size bags."""

    def __init__(
        self,
        cache_path: Path,
        clips_per_recording: int = 8,
        train: bool = False,
        seed: int = 42,
    ):
        payload = torch.load(cache_path, map_location="cpu")
        self.features = payload["features"]
        self.labels = payload["labels"].long()
        self.entries = payload.get("entries", [])
        self.clips_per_recording = int(clips_per_recording)
        self.train = bool(train)
        self.rng = random.Random(seed)

        grouped = defaultdict(list)
        for index, entry in enumerate(self.entries):
            rid = entry.get("recording_id") or Path(entry.get("path", str(index))).stem
            grouped[str(rid)].append(index)

        self.recordings = []
        for rid, indices in grouped.items():
            indices = sorted(
                indices,
                key=lambda i: int(self.entries[i].get("segment_index", i))
                if i < len(self.entries)
                else i,
            )
            labels = {int(self.labels[i]) for i in indices}
            if len(labels) != 1:
                raise ValueError(f"Recording {rid} has inconsistent labels: {sorted(labels)}")
            self.recordings.append((rid, indices, labels.pop()))

        self.recordings.sort(key=lambda item: item[0])

    def __len__(self):
        return len(self.recordings)

    def _sample_indices(self, indices, eval_sample_id: int = 0, eval_samples: int = 1):
        s = self.clips_per_recording
        n = len(indices)
        if n == 0:
            raise ValueError("Empty recording bag")

        if self.train:
            if n >= s:
                picked = self.rng.sample(indices, s)
                return sorted(picked, key=lambda i: int(self.entries[i].get("segment_index", i)))
            return [self.rng.choice(indices) for _ in range(s)]

        eval_samples = max(int(eval_samples), 1)
        eval_sample_id = int(eval_sample_id) % eval_samples
        if n >= s:
            if eval_samples > 1:
                # Deterministic multi-view sampling: each view picks a different
                # phase inside evenly spaced temporal bins.
                phase = (eval_sample_id + 0.5) / eval_samples
                positions = (np.arange(s, dtype=np.float64) + phase) * n / s - 0.5
                positions = np.clip(np.rint(positions), 0, n - 1).astype(int)
                return [indices[int(pos)] for pos in positions]
            positions = np.linspace(0, n - 1, num=s)
            return [indices[int(round(pos))] for pos in positions]
        return [indices[(i + eval_sample_id) % n] for i in range(s)]

    def get_eval_item(self, index, eval_sample_id: int = 0, eval_samples: int = 1):
        rid, indices, label = self.recordings[index]
        picked = self._sample_indices(indices, eval_sample_id=eval_sample_id, eval_samples=eval_samples)
        clips = self.features[picked].float()
        return clips, torch.tensor(label, dtype=torch.long), rid

    def __getitem__(self, index):
        if not self.train:
            return self.get_eval_item(index)
        rid, indices, label = self.recordings[index]
        picked = self._sample_indices(indices)
        clips = self.features[picked].float()
        return clips, torch.tensor(label, dtype=torch.long), rid


class FrozenShuffleFACEncoder(nn.Module):
    def __init__(self, checkpoint_path: Path, device: torch.device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.cnn_cfg = checkpoint["cnn_cfg"]
        self.feats_cfg = checkpoint.get("feats_cfg", {})
        self.best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -1.0))
        self.best_epoch = int(checkpoint.get("epoch", -1))
        self.encoder = shuffleFAC(**self.cnn_cfg).to(device)
        self.encoder.load_state_dict(checkpoint["model_state"])
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    @property
    def embed_dim(self):
        return int(self.cnn_cfg["nb_filters"][-1])

    @torch.no_grad()
    def forward(self, clips):
        self.encoder.eval()
        x = clips.transpose(2, 3)
        emb = self.encoder.cnn(x).view(x.size(0), -1)
        return emb.detach()


class AttentionAggregationHead(nn.Module):
    def __init__(self, encoder: FrozenShuffleFACEncoder, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        dim = encoder.embed_dim
        self.norm = nn.LayerNorm(dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
        self.last_attn_entropy = torch.tensor(0.0)

    def encode_nodes(self, x):
        b, s, c, f, t = x.shape
        flat = x.reshape(b * s, c, f, t)
        emb = self.encoder(flat).view(b, s, -1)
        return self.norm(emb)

    def pool(self, nodes):
        scores = self.attn_mlp(nodes)
        weights = torch.softmax(scores, dim=1)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()
        return (weights * nodes).sum(dim=1)

    def forward(self, x):
        nodes = self.encode_nodes(x)
        recording_emb = self.pool(nodes)
        return self.classifier(recording_emb)


class ClipGraphConv(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, knn_idx):
        b, s, d = x.shape
        k = knn_idx.size(-1)
        expanded = x.unsqueeze(1).expand(b, s, s, d)
        gather_idx = knn_idx.unsqueeze(-1).expand(b, s, k, d)
        neighbors = torch.gather(expanded, dim=2, index=gather_idx)
        delta = (neighbors - x.unsqueeze(2)).mean(dim=2)
        out = self.msg_mlp(torch.cat([x, delta], dim=-1))
        return self.norm(out)


class GraphAggregationHead(AttentionAggregationHead):
    def __init__(
        self,
        encoder: FrozenShuffleFACEncoder,
        num_classes: int,
        graph_k: int = 2,
        edge_mode: str = "temporal_similarity",
        dropout: float = 0.2,
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, dropout=dropout)
        dim = encoder.embed_dim
        self.graph_k = int(graph_k)
        self.edge_mode = edge_mode
        self.graph_conv = ClipGraphConv(dim=dim, dropout=dropout)
        self.graph_res_scale = nn.Parameter(torch.tensor(0.1))
        self.last_graph_delta_norm = torch.tensor(0.0)

    def _build_graph(self, nodes):
        b, s, _ = nodes.shape
        if s <= 1:
            return torch.zeros((b, s, 1), dtype=torch.long, device=nodes.device)

        adj = torch.zeros((b, s, s), dtype=torch.bool, device=nodes.device)
        if self.edge_mode in ("temporal", "temporal_similarity"):
            for i in range(s):
                if i > 0:
                    adj[:, i, i - 1] = True
                if i + 1 < s:
                    adj[:, i, i + 1] = True

        if self.edge_mode in ("similarity", "temporal_similarity"):
            k = min(self.graph_k, s - 1)
            normed = F.normalize(nodes, p=2, dim=-1)
            sim = torch.bmm(normed, normed.transpose(1, 2))
            eye = torch.eye(s, dtype=torch.bool, device=nodes.device).unsqueeze(0)
            sim = sim.masked_fill(eye, -float("inf"))
            sim_idx = sim.topk(k=k, dim=-1).indices
            adj.scatter_(2, sim_idx, True)

        max_degree = int(adj.sum(dim=-1).max().item())
        max_degree = max(max_degree, 1)
        out = torch.zeros((b, s, max_degree), dtype=torch.long, device=nodes.device)
        for bi in range(b):
            for i in range(s):
                idx = torch.nonzero(adj[bi, i], as_tuple=False).flatten()
                if idx.numel() == 0:
                    idx = torch.tensor([i], dtype=torch.long, device=nodes.device)
                if idx.numel() < max_degree:
                    pad = idx[:1].expand(max_degree - idx.numel())
                    idx = torch.cat([idx, pad], dim=0)
                out[bi, i] = idx[:max_degree]
        return out

    def forward(self, x):
        nodes = self.encode_nodes(x)
        if nodes.size(1) <= 1:
            graph_nodes = nodes
        else:
            knn_idx = self._build_graph(nodes)
            graph_update = self.graph_conv(nodes, knn_idx)
            graph_nodes = nodes + self.graph_res_scale * graph_update
        self.last_graph_delta_norm = (graph_nodes - nodes).norm(dim=-1).mean().detach()
        recording_emb = self.pool(graph_nodes)
        return self.classifier(recording_emb)


class GraphAwareAttentionHead(GraphAggregationHead):
    """Uses graph context only for attention weights, then pools original nodes."""

    def __init__(
        self,
        encoder: FrozenShuffleFACEncoder,
        num_classes: int,
        graph_k: int = 2,
        edge_mode: str = "temporal_similarity",
        dropout: float = 0.2,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            graph_k=graph_k,
            edge_mode=edge_mode,
            dropout=dropout,
        )
        dim = encoder.embed_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        nodes = self.encode_nodes(x)
        if nodes.size(1) <= 1:
            graph_context = torch.zeros_like(nodes)
        else:
            knn_idx = self._build_graph(nodes)
            graph_update = self.graph_conv(nodes, knn_idx)
            graph_context = self.graph_res_scale * graph_update
        self.last_graph_delta_norm = graph_context.norm(dim=-1).mean().detach()

        scores = self.attn_mlp(torch.cat([nodes, graph_context], dim=-1))
        weights = torch.softmax(scores, dim=1)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()

        recording_emb = (weights * nodes).sum(dim=1)
        return self.classifier(recording_emb)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def metrics_from_logits(y_true, logits):
    y_true = np.asarray(y_true)
    pred = np.asarray(logits).argmax(axis=1)
    return {
        "ACC": accuracy_score(y_true, pred),
        "Macro-F1": f1_score(y_true, pred, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


@torch.no_grad()
def run_multisample_eval(model, dataset, batch_size, criterion, device, eval_samples: int):
    model.eval()
    model.encoder.encoder.eval()

    losses = []
    y_true = []
    logits_all = []
    entropy_vals = []
    delta_vals = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        labels = torch.stack([dataset.get_eval_item(i)[1] for i in indices]).to(device, non_blocking=True)
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
            if hasattr(model, "last_graph_delta_norm"):
                sample_delta.append(float(model.last_graph_delta_norm.detach().cpu()))
        logits = torch.stack(sample_logits, dim=0).mean(dim=0)
        loss = criterion(logits, labels)

        batch = labels.size(0)
        losses.append(float(loss.detach().cpu()) * batch)
        y_true.extend(labels.detach().cpu().tolist())
        logits_all.append(logits.detach().cpu())
        entropy_vals.append(float(np.mean(sample_entropy)))
        if sample_delta:
            delta_vals.append(float(np.mean(sample_delta)))

    logits_np = torch.cat(logits_all, dim=0).numpy()
    metrics = metrics_from_logits(y_true, logits_np)
    metrics["loss"] = sum(losses) / max(len(y_true), 1)
    metrics["attn_entropy"] = float(np.mean(entropy_vals)) if entropy_vals else math.nan
    if delta_vals:
        metrics["graph_delta_norm"] = float(np.mean(delta_vals))
        metrics["graph_res_scale"] = float(model.graph_res_scale.detach().cpu())
    metrics["eval_samples"] = int(eval_samples)
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None, eval_samples: int = 1):
    train = optimizer is not None
    if (not train) and eval_samples > 1:
        dataset = getattr(loader, "dataset", None)
        if isinstance(dataset, RecordingBagCachedDataset):
            return run_multisample_eval(
                model=model,
                dataset=dataset,
                batch_size=loader.batch_size or 1,
                criterion=criterion,
                device=device,
                eval_samples=eval_samples,
            )

    model.train(mode=train)
    model.encoder.encoder.eval()

    losses = []
    y_true = []
    logits_all = []
    entropy_vals = []
    delta_vals = []
    for clips, labels, _rids in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(clips)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()

        batch = labels.size(0)
        losses.append(float(loss.detach().cpu()) * batch)
        y_true.extend(labels.detach().cpu().tolist())
        logits_all.append(logits.detach().cpu())
        entropy_vals.append(float(model.last_attn_entropy.detach().cpu()))
        if hasattr(model, "last_graph_delta_norm"):
            delta_vals.append(float(model.last_graph_delta_norm.detach().cpu()))

    logits_np = torch.cat(logits_all, dim=0).numpy()
    metrics = metrics_from_logits(y_true, logits_np)
    metrics["loss"] = sum(losses) / max(len(y_true), 1)
    metrics["attn_entropy"] = float(np.mean(entropy_vals)) if entropy_vals else math.nan
    if delta_vals:
        metrics["graph_delta_norm"] = float(np.mean(delta_vals))
        metrics["graph_res_scale"] = float(model.graph_res_scale.detach().cpu())
    metrics["eval_samples"] = 1
    return metrics


def write_metrics(path: Path, rows: dict):
    with path.open("w", encoding="utf-8") as f:
        for key, value in rows.items():
            f.write(f"{key}: {value}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen ShuffleFAC recording-level graph head sanity.")
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--head_type", choices=["attention", "graph", "graph_aware_attention"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clips_per_recording", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph_k", type=int, default=2)
    parser.add_argument(
        "--edge_mode",
        choices=["temporal", "similarity", "temporal_similarity"],
        default="temporal_similarity",
    )
    parser.add_argument("--eval_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_run_log(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    with Path(args.model_config).open("r", encoding="utf-8") as f:
        config = json.load(f)
    cache_paths = config["cache_paths"]
    train_cache = resolve_path(cache_paths["train"], root)
    val_cache = resolve_path(cache_paths["val"], root)
    test_cache = resolve_path(cache_paths["test"], root)

    train_set = RecordingBagCachedDataset(train_cache, args.clips_per_recording, train=True, seed=args.seed)
    val_set = RecordingBagCachedDataset(val_cache, args.clips_per_recording, train=False, seed=args.seed)
    test_set = RecordingBagCachedDataset(test_cache, args.clips_per_recording, train=False, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = FrozenShuffleFACEncoder(resolve_path(args.encoder_ckpt, root), device=device)
    num_classes = int(encoder.cnn_cfg["n_class"])
    if args.head_type == "attention":
        model = AttentionAggregationHead(encoder, num_classes=num_classes, dropout=args.dropout).to(device)
    elif args.head_type == "graph":
        model = GraphAggregationHead(
            encoder,
            num_classes=num_classes,
            graph_k=args.graph_k,
            edge_mode=args.edge_mode,
            dropout=args.dropout,
        ).to(device)
    else:
        model = GraphAwareAttentionHead(
            encoder,
            num_classes=num_classes,
            graph_k=args.graph_k,
            edge_mode=args.edge_mode,
            dropout=args.dropout,
        ).to(device)

    trainable_params, frozen_params = count_params(model)
    print(f"Loaded encoder: {args.encoder_ckpt}", flush=True)
    print(f"Encoder best val Macro-F1: {encoder.best_val_macro_f1:.4f} at epoch {encoder.best_epoch}", flush=True)
    print(f"Encoder frozen params: {frozen_params}", flush=True)
    print(f"Head trainable params: {trainable_params}", flush=True)
    print(f"Recordings train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = -1.0
    best_epoch = -1
    stale_epochs = 0
    best_path = output_dir / "best_head.pt"
    epoch_csv = output_dir / "epoch_metrics.csv"
    with epoch_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_weighted_f1",
                "attn_entropy",
                "graph_delta_norm",
                "graph_res_scale",
            ]
        )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            eval_samples=args.eval_samples,
        )
        graph_delta = val_metrics.get("graph_delta_norm", math.nan)
        graph_res = val_metrics.get("graph_res_scale", math.nan)
        print(
            f"{epoch},{train_metrics['loss']:.6f},{val_metrics['loss']:.6f},"
            f"{val_metrics['ACC']:.6f},{val_metrics['Macro-F1']:.6f},"
            f"{val_metrics['Weighted-F1']:.6f},attn={val_metrics['attn_entropy']:.6f},"
            f"delta={graph_delta:.6f},res={graph_res:.6f}",
            flush=True,
        )
        with epoch_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["ACC"],
                    val_metrics["Macro-F1"],
                    val_metrics["Weighted-F1"],
                    val_metrics["attn_entropy"],
                    graph_delta,
                    graph_res,
                ]
            )

        if val_metrics["Macro-F1"] > best_val:
            best_val = val_metrics["Macro-F1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "head_type": args.head_type,
                    "model_state": model.state_dict(),
                    "best_val_macro_f1": best_val,
                    "args": vars(args),
                    "trainable_params": trainable_params,
                    "frozen_params": frozen_params,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        optimizer=None,
        eval_samples=args.eval_samples,
    )
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        eval_samples=args.eval_samples,
    )

    summary = {
        "Head": args.head_type,
        "Encoder checkpoint": str(resolve_path(args.encoder_ckpt, root)),
        "Encoder loaded": True,
        "Encoder frozen": True,
        "Encoder best val Macro-F1": encoder.best_val_macro_f1,
        "Encoder best epoch": encoder.best_epoch,
        "Best Val Macro-F1": best_val,
        "Best epoch": best_epoch,
        "Val ACC": val_metrics["ACC"],
        "Val Macro-F1": val_metrics["Macro-F1"],
        "Val Weighted-F1": val_metrics["Weighted-F1"],
        "Test Recording ACC": test_metrics["ACC"],
        "Test Recording Macro-F1": test_metrics["Macro-F1"],
        "Test Recording Weighted-F1": test_metrics["Weighted-F1"],
        "eval_samples": args.eval_samples,
        "eval_sampling": "deterministic_multisample" if args.eval_samples > 1 else "deterministic_even",
        "attn_entropy": test_metrics["attn_entropy"],
        "graph_delta_norm": test_metrics.get("graph_delta_norm", None),
        "graph_res_scale": test_metrics.get("graph_res_scale", None),
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "clips_per_recording": args.clips_per_recording,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "edge_mode": args.edge_mode,
        "seed": args.seed,
    }
    write_metrics(output_dir / "metrics.txt", summary)
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(summary | {"cache_paths": cache_paths}, f, indent=2)

    print("Final metrics", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
