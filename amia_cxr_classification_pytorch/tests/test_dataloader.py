from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import AMIACXRDataset


def test_dataset_returns_expected_shape(tmp_path):
    raw_dir = tmp_path / "raw"
    img_dir = raw_dir / "train" / "train"
    img_dir.mkdir(parents=True)
    arr = (np.random.rand(32, 32) * 255).astype("uint8")
    Image.fromarray(arr).save(img_dir / "img1.png")

    metadata = pd.DataFrame({
        "image_id": ["img1"],
        "image_path": ["train/train/img1.png"],
        "split": ["train"],
        **{f"label_{i}": [1 if i == 0 else 0] for i in range(14)},
    })
    meta_csv = tmp_path / "metadata.csv"
    metadata.to_csv(meta_csv, index=False)

    ds = AMIACXRDataset(
        raw_dir=raw_dir,
        metadata_csv=meta_csv,
        split="train",
        image_size=64,
        input_channels=1,
        normalization="none",
        augmentation=False,
    )
    sample = ds[0]
    assert sample["image"].shape == (1, 64, 64)
    assert sample["label"].shape[0] == 14
    assert sample["has_label"].item() is True
