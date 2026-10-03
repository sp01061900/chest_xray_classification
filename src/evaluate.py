"""Evaluate a trained model or export probabilities for unlabeled test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from constants import CLASS_NAMES, NUM_CLASSES
from datasets import build_dataloader
from models import build_model
from utils import ensure_dir, get_device, load_config, save_json


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def per_class_metrics(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            "class_id": i,
            "class_name": name,
            "roc_auc": safe_metric(roc_auc_score, y_true[:, i], probs[:, i]),
            "pr_auc": safe_metric(average_precision_score, y_true[:, i], probs[:, i]),
            "f1": safe_metric(f1_score, y_true[:, i], preds[:, i], zero_division=0),
            "precision": safe_metric(precision_score, y_true[:, i], preds[:, i], zero_division=0),
            "recall": safe_metric(recall_score, y_true[:, i], preds[:, i], zero_division=0),
            "support": int(y_true[:, i].sum()),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    return {
        "macro_roc_auc": safe_metric(roc_auc_score, y_true, probs, average="macro"),
        "micro_roc_auc": safe_metric(roc_auc_score, y_true, probs, average="micro"),
        "macro_pr_auc": safe_metric(average_precision_score, y_true, probs, average="macro"),
        "micro_pr_auc": safe_metric(average_precision_score, y_true, probs, average="micro"),
        "macro_f1": safe_metric(f1_score, y_true, preds, average="macro", zero_division=0),
        "micro_f1": safe_metric(f1_score, y_true, preds, average="micro", zero_division=0),
        "macro_precision": safe_metric(precision_score, y_true, preds, average="macro", zero_division=0),
        "macro_recall": safe_metric(recall_score, y_true, preds, average="macro", zero_division=0),
    }


def load_model_from_run(run_dir: Path, checkpoint_name: str, device: torch.device):
    config = load_config(run_dir / "config.yaml")
    model = build_model(config).to(device)
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def predict(model, loader, device: torch.device, amp: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    all_logits, all_labels, all_has_labels, image_ids = [], [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
        all_has_labels.append(batch["has_label"].detach().cpu())
        image_ids.extend(batch["image_id"])

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    has_labels = torch.cat(all_has_labels).numpy().astype(bool)
    return logits, labels, has_labels, image_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AMIA CXR classifier.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--thresholds", default=None, type=Path)
    args = parser.parse_args()

    device = get_device()
    model, config, _ = load_model_from_run(args.run_dir, args.checkpoint, device)
    config["preprocessing"]["augmentation"] = False
    loader = build_dataloader(config, split=args.split, train=False)

    amp = bool(config["training"].get("amp", True))
    logits, labels, has_labels, image_ids = predict(model, loader, device, amp)
    probs = sigmoid(logits)

    metrics_dir = ensure_dir(config["output"].get("metrics_dir", "outputs/metrics"))
    exp_name = config.get("experiment_name", config["model"]["name"])

    pred_df = pd.DataFrame({"image_id": image_ids})
    for i, name in enumerate(CLASS_NAMES):
        safe_name = name.replace("/", "_").replace(" ", "_")
        pred_df[f"prob_{i}_{safe_name}"] = probs[:, i]
    pred_path = metrics_dir / f"{exp_name}_{args.split}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    if not has_labels.all():
        print(f"Split '{args.split}' appears unlabeled. Saved probability predictions to {pred_path}")
        return

    if args.thresholds and args.thresholds.exists():
        thresholds = np.load(args.thresholds).astype(np.float32)
    elif (args.run_dir / "thresholds.npy").exists():
        thresholds = np.load(args.run_dir / "thresholds.npy").astype(np.float32)
    else:
        thresholds = np.full(NUM_CLASSES, float(config["evaluation"].get("default_threshold", 0.5)), dtype=np.float32)

    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    metrics = aggregate_metrics(labels, probs, thresholds)
    metrics["split"] = args.split
    metrics["checkpoint"] = args.checkpoint
    metrics["num_samples"] = int(labels.shape[0])

    per_cls = per_class_metrics(labels, probs, preds)
    per_cls.to_csv(metrics_dir / f"{exp_name}_{args.split}_per_class_metrics.csv", index=False)
    save_json(metrics, metrics_dir / f"{exp_name}_{args.split}_metrics.json")
    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {pred_path}")


if __name__ == "__main__":
    main()
