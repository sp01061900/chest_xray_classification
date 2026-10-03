"""PyTorch Dataset and DataLoader helpers for AMIA chest X-ray classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from constants import NUM_CLASSES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class XRayTensorNormalize:
    """TorchXRayVision-style normalization from [0, 1] to roughly [-1024, 1024]."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2048.0 - 1024.0


class AMIACXRDataset(Dataset):
    """Dataset for image-level multi-label chest X-ray classification."""

    def __init__(
        self,
        raw_dir: str | Path,
        metadata_csv: str | Path,
        split: str,
        image_size: int = 224,
        input_channels: int = 1,
        normalization: str = "xrv",
        augmentation: bool = False,
        stats_mean: Optional[str | Path] = None,
        stats_std: Optional[str | Path] = None,
        use_cache: bool = False,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.split = split
        self.image_size = image_size
        self.input_channels = input_channels
        self.normalization = normalization
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.label_cols = [f"label_{i}" for i in range(NUM_CLASSES)]

        self.transform = self._build_transform(augmentation, stats_mean, stats_std)

    def __len__(self) -> int:
        return len(self.metadata)

    def _build_transform(self, augmentation: bool, stats_mean: Optional[str | Path], stats_std: Optional[str | Path]):
        ops = []
        if augmentation:
            ops.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            ])
        ops.extend([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        if self.normalization == "imagenet":
            ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        elif self.normalization == "stats":
            if stats_mean is None or stats_std is None:
                raise ValueError("stats normalization requires stats_mean and stats_std paths")
            mean = np.load(stats_mean).astype(float).tolist()
            std = np.load(stats_std).astype(float).tolist()
            if self.input_channels == 3 and len(mean) == 1:
                mean = mean * 3
                std = std * 3
            ops.append(transforms.Normalize(mean, std))
        elif self.normalization == "xrv":
            ops.append(XRayTensorNormalize())
        elif self.normalization in ("none", None):
            pass
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        return transforms.Compose(ops)

    def _load_from_cache(self, image_id: str) -> Optional[Image.Image]:
        if not self.use_cache or self.cache_dir is None:
            return None
        npy_path = self.cache_dir / f"{image_id}.npy"
        if not npy_path.exists():
            return None
        arr = np.load(npy_path)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def _load_image(self, row: pd.Series) -> Image.Image:
        image_id = str(row["image_id"])
        cached = self._load_from_cache(image_id)
        if cached is not None:
            img = cached
        else:
            image_path = self.raw_dir / row["image_path"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            img = Image.open(image_path).convert("L")

        if self.input_channels == 3:
            img = img.convert("RGB")
        elif self.input_channels == 1:
            img = img.convert("L")
        else:
            raise ValueError("input_channels must be 1 or 3")
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.metadata.iloc[idx]
        img = self._load_image(row)
        image = self.transform(img).float()

        labels = row[self.label_cols].values.astype(np.float32)
        has_labels = not np.isnan(labels).any()
        if not has_labels:
            labels = np.zeros(NUM_CLASSES, dtype=np.float32)

        return {
            "image": image,
            "label": torch.tensor(labels, dtype=torch.float32),
            "image_id": str(row["image_id"]),
            "has_label": torch.tensor(has_labels, dtype=torch.bool),
        }


def build_dataloader(config: Dict, split: str, train: bool) -> DataLoader:
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    dataset = AMIACXRDataset(
        raw_dir=data_cfg["raw_dir"],
        metadata_csv=data_cfg["metadata_csv"],
        split=split,
        image_size=prep_cfg.get("image_size", 224),
        input_channels=model_cfg.get("input_channels", 1),
        normalization=prep_cfg.get("normalization", "xrv"),
        augmentation=bool(prep_cfg.get("augmentation", False)) and train,
        stats_mean=data_cfg.get("stats_mean"),
        stats_std=data_cfg.get("stats_std"),
        use_cache=bool(data_cfg.get("use_cache", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )

    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=train,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
