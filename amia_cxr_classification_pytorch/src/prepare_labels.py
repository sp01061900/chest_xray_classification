"""Prepare image-level multi-label targets for AMIA chest X-ray classification.

This script converts row-level annotations from train.csv into a single 14-element
multi-hot vector per image. Bounding boxes are ignored because this repository is
classification-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constants import CLASS_NAMES, IMAGE_EXTENSIONS, NO_FINDING_CLASS_ID, NUM_CLASSES


def find_image_file(root: Path, image_id: str) -> str:
    """Return image path relative to root for a given image_id."""
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(root.rglob(f"{image_id}{ext}"))
    if not candidates:
        return ""
    # Prefer train/train or test/test exact nested structure if present.
    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return str(candidates[0].relative_to(root))


def build_multilabel_table(train_csv: str | Path) -> pd.DataFrame:
    """Aggregate train.csv rows into image-level labels.

    Returns a DataFrame with one row per image and columns:
    image_id, label_0, ..., label_13.
    """
    df = pd.read_csv(train_csv)
    required = {"image_id", "class_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce").fillna(NO_FINDING_CLASS_ID).astype(int)

    rows = []
    for image_id, group in df.groupby("image_id", sort=True):
        labels = np.zeros(NUM_CLASSES, dtype=np.float32)
        for class_id in group["class_id"].tolist():
            if 0 <= class_id < NUM_CLASSES:
                labels[class_id] = 1.0
            elif class_id == NO_FINDING_CLASS_ID:
                # No finding is represented as all-zero abnormality labels.
                continue
            else:
                raise ValueError(f"Unexpected class_id={class_id} for image_id={image_id}")
        row = {"image_id": image_id}
        row.update({f"label_{i}": int(labels[i]) for i in range(NUM_CLASSES)})
        rows.append(row)

    return pd.DataFrame(rows)


def discover_test_images(raw_dir: Path) -> pd.DataFrame:
    """Discover test image IDs and relative image paths."""
    test_root = raw_dir / "test"
    if not test_root.exists():
        return pd.DataFrame(columns=["image_id", "image_path", "split"])

    records = []
    for path in sorted(test_root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append({
                "image_id": path.stem,
                "image_path": str(path.relative_to(raw_dir)),
                "split": "test",
            })
    return pd.DataFrame(records)


def add_image_paths(raw_dir: Path, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Attach relative image paths to each labeled image."""
    paths = []
    missing = []
    for image_id in labels_df["image_id"].astype(str):
        rel = find_image_file(raw_dir / "train", image_id)
        if not rel:
            rel = find_image_file(raw_dir, image_id)
        if rel:
            # If found relative to raw_dir/train, normalize to raw_dir-relative path.
            p = Path(rel)
            candidate = raw_dir / "train" / p
            if candidate.exists():
                rel = str(candidate.relative_to(raw_dir))
        else:
            missing.append(image_id)
        paths.append(rel)

    out = labels_df.copy()
    out["image_path"] = paths
    out = out[out["image_path"] != ""].reset_index(drop=True)

    if missing:
        print(f"Warning: {len(missing)} labeled images were not found on disk and were skipped.")
    return out


def make_train_val_split(labels_df: pd.DataFrame, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic image-level train/validation split.

    Multi-label iterative stratification is intentionally not required as a dependency.
    Instead, we stratify only on whether an image has any abnormal finding. This keeps
    normal/abnormal balance roughly stable while remaining lightweight.
    """
    label_cols = [f"label_{i}" for i in range(NUM_CLASSES)]
    any_positive = labels_df[label_cols].sum(axis=1).clip(upper=1).astype(int)
    image_ids = labels_df["image_id"].values

    try:
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=any_positive,
        )
    except ValueError:
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return np.array(train_ids), np.array(val_ids)


def compute_pos_weight(train_labels: np.ndarray) -> np.ndarray:
    """Compute class-wise positive weights for BCEWithLogitsLoss."""
    positives = train_labels.sum(axis=0)
    negatives = train_labels.shape[0] - positives
    # Avoid division by zero and extreme values.
    pos_weight = negatives / np.clip(positives, 1.0, None)
    return np.clip(pos_weight, 1.0, 50.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AMIA multi-label classification metadata.")
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--split_dir", required=True, type=Path)
    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.raw_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Could not find train.csv at {train_csv}")

    labels_df = build_multilabel_table(train_csv)
    labels_df = add_image_paths(args.raw_dir, labels_df)

    train_ids, val_ids = make_train_val_split(labels_df, args.val_size, args.seed)
    labels_df["split"] = "train"
    labels_df.loc[labels_df["image_id"].isin(val_ids), "split"] = "val"

    test_df = discover_test_images(args.raw_dir)
    for i in range(NUM_CLASSES):
        test_df[f"label_{i}"] = np.nan

    metadata = pd.concat([labels_df, test_df], ignore_index=True, sort=False)
    metadata = metadata[["image_id", "image_path", "split"] + [f"label_{i}" for i in range(NUM_CLASSES)]]
    metadata.to_csv(args.out_dir / "metadata.csv", index=False)

    label_cols = [f"label_{i}" for i in range(NUM_CLASSES)]
    train_labels = labels_df[labels_df["split"] == "train"][label_cols].values.astype(np.float32)
    val_labels = labels_df[labels_df["split"] == "val"][label_cols].values.astype(np.float32)

    np.save(args.split_dir / "train_image_ids.npy", train_ids)
    np.save(args.split_dir / "val_image_ids.npy", val_ids)
    if len(test_df) > 0:
        np.save(args.split_dir / "test_image_ids.npy", test_df["image_id"].values)

    np.save(args.out_dir / "labels_train.npy", train_labels)
    np.save(args.out_dir / "labels_val.npy", val_labels)
    np.save(args.out_dir / "pos_weight.npy", compute_pos_weight(train_labels))

    class_counts = pd.DataFrame({
        "class_id": list(range(NUM_CLASSES)),
        "class_name": CLASS_NAMES,
        "train_positive_count": train_labels.sum(axis=0).astype(int),
        "val_positive_count": val_labels.sum(axis=0).astype(int),
    })
    class_counts.to_csv(args.out_dir / "class_counts.csv", index=False)

    with open(args.out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump({str(i): name for i, name in enumerate(CLASS_NAMES)}, f, indent=2)

    print(f"Saved metadata: {args.out_dir / 'metadata.csv'}")
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)} | Test images: {len(test_df)}")
    print(f"Saved class counts and NumPy arrays under: {args.out_dir}")


if __name__ == "__main__":
    main()
