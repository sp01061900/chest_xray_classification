"""Generate comparison and per-class metric plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_training_curves(checkpoints_dir: Path, out_dir: Path) -> None:
    rows = []
    for history_path in checkpoints_dir.glob("*/history.csv"):
        exp = history_path.parent.name
        df = pd.read_csv(history_path)
        if len(df) == 0:
            continue
        df["experiment"] = exp
        rows.append(df)
    if not rows:
        return
    history = pd.concat(rows, ignore_index=True)

    for metric in ["val_loss", "macro_roc_auc", "macro_pr_auc", "macro_f1"]:
        if metric not in history.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=history, x="epoch", y=metric, hue="experiment", marker="o")
        plt.title(f"Validation {metric} by epoch")
        plt.tight_layout()
        plt.savefig(out_dir / f"training_curve_{metric}.png", dpi=200)
        plt.close()


def plot_model_comparison(metrics_dir: Path, out_dir: Path) -> None:
    rows = []
    for path in metrics_dir.glob("*_val_metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["experiment"] = path.name.replace("_val_metrics.json", "")
        rows.append(record)
    if not rows:
        return
    df = pd.DataFrame(rows)
    metric_cols = [c for c in ["macro_roc_auc", "macro_pr_auc", "macro_f1", "micro_f1"] if c in df.columns]
    if not metric_cols:
        return
    long_df = df.melt(id_vars="experiment", value_vars=metric_cols, var_name="metric", value_name="value")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=long_df, x="experiment", y="value", hue="metric")
    plt.xticks(rotation=25, ha="right")
    plt.title("Model comparison on validation split")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_val_metrics.png", dpi=200)
    plt.close()


def plot_per_class(metrics_dir: Path, out_dir: Path) -> None:
    for path in metrics_dir.glob("*_val_per_class_metrics.csv"):
        exp = path.name.replace("_val_per_class_metrics.csv", "")
        df = pd.read_csv(path)
        for metric in ["roc_auc", "pr_auc", "f1"]:
            if metric not in df.columns:
                continue
            plt.figure(figsize=(12, 7))
            sns.barplot(data=df, y="class_name", x=metric, color="#4C78A8")
            plt.title(f"{exp}: per-class {metric}")
            plt.xlim(0, 1)
            plt.tight_layout()
            plt.savefig(out_dir / f"{exp}_per_class_{metric}.png", dpi=200)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AMIA CXR experiment results.")
    parser.add_argument("--checkpoints_dir", default="outputs/checkpoints", type=Path)
    parser.add_argument("--metrics_dir", default="outputs/metrics", type=Path)
    parser.add_argument("--out_dir", default="outputs/plots", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(args.checkpoints_dir, args.out_dir)
    plot_model_comparison(args.metrics_dir, args.out_dir)
    plot_per_class(args.metrics_dir, args.out_dir)
    print(f"Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
