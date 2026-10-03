"""Training entry point for AMIA chest X-ray multi-label classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from constants import CLASS_NAMES, NUM_CLASSES
from datasets import build_dataloader
from models import build_model, freeze_backbone, make_param_groups, unfreeze_all
from utils import AverageMeter, append_jsonl, ensure_dir, get_device, get_logger, load_config, save_checkpoint, save_config, seed_everything


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    """Return NaN instead of failing when a class has only one target value."""
    try:
        return float(fn(y_true, y_score, **kwargs))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    metrics = {
        "macro_roc_auc": safe_metric(roc_auc_score, y_true, probs, average="macro"),
        "micro_roc_auc": safe_metric(roc_auc_score, y_true, probs, average="micro"),
        "macro_pr_auc": safe_metric(average_precision_score, y_true, probs, average="macro"),
        "micro_pr_auc": safe_metric(average_precision_score, y_true, probs, average="micro"),
        "macro_f1": safe_metric(f1_score, y_true, preds, average="macro", zero_division=0),
        "micro_f1": safe_metric(f1_score, y_true, preds, average="micro", zero_division=0),
        "macro_precision": safe_metric(precision_score, y_true, preds, average="macro", zero_division=0),
        "macro_recall": safe_metric(recall_score, y_true, preds, average="macro", zero_division=0),
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool, clip_grad_norm: float | None) -> float:
    model.train()
    meter = AverageMeter()
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if clip_grad_norm is not None and clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool) -> Tuple[float, Dict[str, float]]:
    model.eval()
    meter = AverageMeter()
    all_logits = []
    all_labels = []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        meter.update(loss.item(), images.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    metrics = compute_metrics(labels_np, logits_np, threshold=0.5)
    return meter.avg, metrics


def build_optimizer(model, config):
    train_cfg = config["training"]
    groups = make_param_groups(
        model,
        lr_head=float(train_cfg["lr_head"]),
        lr_backbone=float(train_cfg["lr_backbone"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(groups)
    if opt_name == "adam":
        return torch.optim.Adam(groups)
    if opt_name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AMIA CXR classifier.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["training"].get("seed", 42)))

    exp_name = config.get("experiment_name", config["model"]["name"])
    run_dir = ensure_dir(Path(config["output"]["checkpoints_dir"]) / exp_name)
    metrics_dir = ensure_dir(config["output"]["metrics_dir"])
    logger = get_logger("train", run_dir / "train.log")
    save_config(config, run_dir / "config.yaml")

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {exp_name}")

    train_loader = build_dataloader(config, split="train", train=True)
    val_loader = build_dataloader(config, split="val", train=False)

    model = build_model(config).to(device)

    pos_weight = None
    if config["training"].get("use_pos_weight", True):
        pos_weight_path = Path(config["data"].get("pos_weight", ""))
        if pos_weight_path.exists():
            pos_weight = torch.tensor(np.load(pos_weight_path), dtype=torch.float32, device=device)
            logger.info(f"Loaded pos_weight from {pos_weight_path}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    amp = bool(config["training"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    monitor = config["training"].get("monitor", "macro_roc_auc")
    mode = config["training"].get("monitor_mode", "max")
    best_score = -float("inf") if mode == "max" else float("inf")
    best_epoch = -1
    patience = int(config["training"].get("early_stopping_patience", 5))
    no_improve = 0
    history = []

    total_epochs = int(config["training"].get("epochs_head", 0)) + int(config["training"].get("epochs_finetune", 0))
    epoch_counter = 0

    stages = []
    if int(config["training"].get("epochs_head", 0)) > 0:
        stages.append(("head", int(config["training"]["epochs_head"])))
    if int(config["training"].get("epochs_finetune", 0)) > 0:
        stages.append(("finetune", int(config["training"]["epochs_finetune"])))

    for stage_name, stage_epochs in stages:
        if stage_name == "head":
            freeze_backbone(model)
            logger.info("Stage 1: training classifier head only")
        else:
            unfreeze_all(model)
            logger.info("Stage 2: fine-tuning full model")

        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage_epochs))

        for _ in range(stage_epochs):
            epoch_counter += 1
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                amp=amp,
                clip_grad_norm=config["training"].get("clip_grad_norm", None),
            )
            val_loss, val_metrics = validate(model, val_loader, criterion, device, amp=amp)
            scheduler.step()

            record = {
                "epoch": epoch_counter,
                "stage": stage_name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            history.append(record)
            append_jsonl(record, run_dir / "history.jsonl")
            logger.info(json.dumps(record, indent=2))

            current = val_metrics.get(monitor, float("nan"))
            improved = False
            if np.isfinite(current):
                improved = current > best_score if mode == "max" else current < best_score

            save_checkpoint(run_dir / "last.pth", model, optimizer, epoch_counter, record, config)
            if improved:
                best_score = current
                best_epoch = epoch_counter
                no_improve = 0
                save_checkpoint(run_dir / "best.pth", model, optimizer, epoch_counter, record, config)
                logger.info(f"New best {monitor}: {best_score:.5f} at epoch {best_epoch}")
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Early stopping after {no_improve} non-improving epochs.")
                break

        if no_improve >= patience:
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with open(metrics_dir / f"{exp_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_score": best_score, "monitor": monitor}, f, indent=2)
    logger.info(f"Training complete. Best epoch={best_epoch}, best {monitor}={best_score}")


if __name__ == "__main__":
    main()
