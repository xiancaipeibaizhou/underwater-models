#!/usr/bin/env python
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def parse_value(text):
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_metrics(path):
    metrics = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip()] = parse_value(value)
    return metrics


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def audit_overlap(run_dir):
    audit = load_json(run_dir / "split_audit.txt")
    overlaps = audit.get("recording_overlap", {})
    if not overlaps:
        return "unknown"
    counts = {name: int(data.get("count", 0)) for name, data in overlaps.items()}
    return "0" if all(v == 0 for v in counts.values()) else json.dumps(counts, ensure_ascii=False)


def discover_uatr_rows(root):
    rows = []
    for metrics_path in sorted(root.glob("**/Run_*/metrics.txt")):
        run_dir = metrics_path.parent
        metrics = parse_metrics(metrics_path)
        config = load_json(run_dir / "model_config.json")
        complexity = config.get("complexity", {})
        variant = config.get("uatr_variant") or "C"
        run_match = re.search(r"Run_(\d+)$", run_dir.name)
        run_index = int(run_match.group(1)) if run_match else None
        seed = config.get("training_seed", 42 + run_index if run_index is not None else "unknown")
        rows.append(
            {
                "Model": f"UATR_KNN-{variant}",
                "Seed": seed,
                "Split": "recording-level",
                "Segment": f"{config.get('segment_length', 5)}s",
                "n_fft/hop": f"{config.get('window_length', 2048)}/{config.get('hop_length', 512)}",
                "Overlap": audit_overlap(run_dir),
                "ACC": metrics.get("Test ACC"),
                "Macro-F1": metrics.get("Test F1_Macro"),
                "Weighted-F1": metrics.get("Test F1_Weighted"),
                "Params": metrics.get("Trainable Params", complexity.get("trainable_params")),
                "MACs": metrics.get("MACs formatted", complexity.get("macs_formatted", metrics.get("MACs"))),
                "Latency": metrics.get("Latency ms/batch1", complexity.get("latency_ms_batch1")),
            }
        )
    return rows


def discover_shufflefac_rows(root):
    rows = []
    for metrics_path in sorted(root.glob("**/metrics.txt")):
        if "Run_" in str(metrics_path):
            continue
        run_dir = metrics_path.parent
        metrics = parse_metrics(metrics_path)
        config = load_json(run_dir / "model_config.json")
        args = config.get("training", {})
        split_meta = config.get("split_metadata", {})
        feats = config.get("feats_cfg_used", {})
        gamma_match = re.search(r"gamma=(\d+)", str(metrics.get("Model", "")))
        gamma = gamma_match.group(1) if gamma_match else "16"
        rows.append(
            {
                "Model": f"ShuffleFAC gamma={gamma}",
                "Seed": metrics.get("Seed", args.get("training_seed", "unknown")),
                "Split": split_meta.get("protocol", metrics.get("Split Protocol", "recording-level")),
                "Segment": f"{split_meta.get('segment_length', 5)}s",
                "n_fft/hop": f"{feats.get('n_fft', 2048)}/{feats.get('hop_length', 512)}",
                "Overlap": metrics.get("Recording Overlap?", "unknown"),
                "ACC": metrics.get("ACC"),
                "Macro-F1": metrics.get("Macro-F1"),
                "Weighted-F1": metrics.get("Weighted-F1"),
                "Params": metrics.get("Trainable Params", metrics.get("Params")),
                "MACs": metrics.get("MACs formatted", metrics.get("MACs")),
                "Latency": metrics.get("Latency ms/batch1"),
            }
        )
    return rows


def fmt_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def mean_std(values):
    nums = [numeric(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return "pending"
    mean = statistics.mean(nums)
    std = statistics.stdev(nums) if len(nums) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def summary_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Model"], []).append(row)
    out = []
    for model, items in grouped.items():
        first = items[0]
        out.append(
            {
                "Model": model,
                "ACC mean±std": mean_std([r["ACC"] for r in items]),
                "Macro-F1 mean±std": mean_std([r["Macro-F1"] for r in items]),
                "Weighted-F1 mean±std": mean_std([r["Weighted-F1"] for r in items]),
                "Params": fmt_value(first.get("Params")),
                "MACs": fmt_value(first.get("MACs")),
                "Latency": fmt_value(first.get("Latency")),
            }
        )
    return out


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize UATR_KNN and ShuffleFAC multi-seed fair comparison.")
    parser.add_argument("--uatr_root", type=Path, default=None)
    parser.add_argument("--shufflefac_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results/fair_multiseed_analysis"))
    args = parser.parse_args()

    rows = []
    if args.uatr_root:
        rows.extend(discover_uatr_rows(args.uatr_root))
    if args.shufflefac_root:
        rows.extend(discover_shufflefac_rows(args.shufflefac_root))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_cols = [
        "Model",
        "Seed",
        "Split",
        "Segment",
        "n_fft/hop",
        "Overlap",
        "ACC",
        "Macro-F1",
        "Weighted-F1",
        "Params",
        "MACs",
        "Latency",
    ]
    summary_cols = [
        "Model",
        "ACC mean±std",
        "Macro-F1 mean±std",
        "Weighted-F1 mean±std",
        "Params",
        "MACs",
        "Latency",
    ]

    summary = summary_rows(rows)
    md = "# Fair Multi-Seed Analysis\n\n"
    md += "## Per-Seed Results\n\n"
    md += markdown_table(rows, detailed_cols)
    md += "\n\n## Summary\n\n"
    md += markdown_table(summary, summary_cols)
    md += "\n"
    (args.output_dir / "fair_multiseed_summary.md").write_text(md, encoding="utf-8")

    with (args.output_dir / "fair_multiseed_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt_value(row.get(col)) for col in detailed_cols})
    with (args.output_dir / "fair_multiseed_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for row in summary:
            writer.writerow({col: fmt_value(row.get(col)) for col in summary_cols})

    print(args.output_dir / "fair_multiseed_summary.md")


if __name__ == "__main__":
    main()
