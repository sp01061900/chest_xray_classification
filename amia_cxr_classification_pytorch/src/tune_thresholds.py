"""Tune per-class thresholds on the validation split for multi-label classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from constants import CLASS_NAMES, NUM_CLASSES
from datasets import build_dataloader
from evaluate import load_model_from_run, predict, sigmoid
from utils import get_device, load_config


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: list[float]) -> tuple[np.ndarray, pd.DataFrame]:
    thresholds = np.zeros(NUM_CLASSES, dtype=np.float32)
    rows = []
    for i, class_name in enumerate(CLASS_NAMES):
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            preds = (probs[:, i] >= t).astype(int)
            score = f1_score(y_true[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresholds[i] = best_t
        rows.append({"class_id": i, "class_name": class_name, "threshold": best_t, "val_f1": best_f1})
    return thresholds, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune validation thresholds for AMIA CXR classifier.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pth")
    args = parser.parse_args()

    device = get_device()
    model, config, _ = load_model_from_run(args.run_dir, args.checkpoint, device)
    config["preprocessing"]["augmentation"] = False
    loader = build_dataloader(config, split="val", train=False)
    amp = bool(config["training"].get("amp", True))

    logits, labels, has_labels, image_ids = predict(model, loader, device, amp)
    if not has_labels.all():
        raise RuntimeError("Validation split has missing labels; cannot tune thresholds.")
    probs = sigmoid(logits)

    grid = config.get("evaluation", {}).get("threshold_grid", [round(x, 2) for x in np.linspace(0.05, 0.95, 19)])
    thresholds, table = tune_thresholds(labels, probs, grid)

    np.save(args.run_dir / "thresholds.npy", thresholds)
    table.to_csv(args.run_dir / "thresholds.csv", index=False)
    with open(args.run_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({str(i): float(t) for i, t in enumerate(thresholds)}, f, indent=2)

    print(table.to_string(index=False))
    print(f"Saved thresholds to {args.run_dir / 'thresholds.npy'}")


if __name__ == "__main__":
    main()
