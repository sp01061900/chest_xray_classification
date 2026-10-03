"""Compute preprocessing statistics and optionally cache resized images as NumPy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_grayscale_image(path: Path, img_size: int) -> np.ndarray:
    """Load image as grayscale float32 array in [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_mean_std(raw_dir: Path, metadata: pd.DataFrame, img_size: int, max_images: int | None) -> Tuple[float, float]:
    """Compute dataset mean/std using NumPy over training split."""
    train_df = metadata[metadata["split"] == "train"].copy()
    if max_images is not None and len(train_df) > max_images:
        train_df = train_df.sample(max_images, random_state=42)

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing mean/std"):
        path = raw_dir / row["image_path"]
        if not path.exists():
            continue
        arr = load_grayscale_image(path, img_size)
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float((arr ** 2).sum())
        pixel_count += arr.size

    if pixel_count == 0:
        raise RuntimeError("No training images were found for preprocessing statistics.")

    mean = pixel_sum / pixel_count
    var = max(pixel_sq_sum / pixel_count - mean ** 2, 1e-8)
    std = float(np.sqrt(var))
    return float(mean), std


def cache_images(raw_dir: Path, metadata: pd.DataFrame, out_dir: Path, img_size: int) -> int:
    """Save resized grayscale images as .npy arrays for faster repeated experiments."""
    cache_dir = out_dir / f"cache_{img_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Caching images"):
        image_path = raw_dir / row["image_path"]
        if not image_path.exists():
            continue
        arr = load_grayscale_image(image_path, img_size)
        np.save(cache_dir / f"{row['image_id']}.npy", arr.astype(np.float32))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute image statistics and optional NumPy cache.")
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--max_images", default=None, type=int)
    parser.add_argument("--cache_images", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)

    mean, std = compute_mean_std(args.raw_dir, metadata, args.img_size, args.max_images)
    np.save(args.out_dir / "mean.npy", np.array([mean], dtype=np.float32))
    np.save(args.out_dir / "std.npy", np.array([std], dtype=np.float32))

    cached = 0
    if args.cache_images:
        cached = cache_images(args.raw_dir, metadata, args.out_dir, args.img_size)

    summary = {
        "image_size": args.img_size,
        "mean": mean,
        "std": std,
        "cached_images": cached,
        "cache_dir": str(args.out_dir / f"cache_{args.img_size}") if args.cache_images else None,
    }
    with open(args.out_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
