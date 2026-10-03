"""Simple Grad-CAM visualization for classification-only chest X-ray models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets import AMIACXRDataset
from evaluate import load_model_from_run
from utils import get_device


def find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found; Grad-CAM is not available for this model.")
    return last_conv


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probs = torch.sigmoid(logits)
        if target_class is None:
            target_class = int(probs[0].argmax().item())
        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, probs[0].detach().cpu().numpy()

    def close(self) -> None:
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def denormalize_for_display(tensor: torch.Tensor, normalization: str) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
        if normalization == "imagenet":
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = arr * std + mean
        arr = np.clip(arr, 0, 1)
        arr = arr.mean(axis=2)
    else:
        arr = arr[0]
        if normalization == "xrv":
            arr = (arr + 1024.0) / 2048.0
        arr = np.clip(arr, 0, 1)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a trained model.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--target_class", default=None, type=int)
    parser.add_argument("--out_dir", default="outputs/plots/gradcam", type=Path)
    args = parser.parse_args()

    device = get_device()
    model, config, _ = load_model_from_run(args.run_dir, args.checkpoint, device)
    config["preprocessing"]["augmentation"] = False
    dataset = AMIACXRDataset(
        raw_dir=config["data"]["raw_dir"],
        metadata_csv=config["data"]["metadata_csv"],
        split=args.split,
        image_size=config["preprocessing"].get("image_size", 224),
        input_channels=config["model"].get("input_channels", 1),
        normalization=config["preprocessing"].get("normalization", "xrv"),
        augmentation=False,
        stats_mean=config["data"].get("stats_mean"),
        stats_std=config["data"].get("stats_std"),
        use_cache=config["data"].get("use_cache", False),
        cache_dir=config["data"].get("cache_dir"),
    )

    matches = dataset.metadata.index[dataset.metadata["image_id"].astype(str) == str(args.image_id)].tolist()
    if not matches:
        raise ValueError(f"image_id={args.image_id} not found in split={args.split}")
    sample = dataset[matches[0]]
    image = sample["image"].unsqueeze(0).to(device)

    target_layer = find_last_conv(model)
    cam_engine = GradCAM(model, target_layer)
    cam, target_class, probs = cam_engine(image, args.target_class)
    cam_engine.close()

    base = denormalize_for_display(sample["image"], config["preprocessing"].get("normalization", "xrv"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"gradcam_{args.image_id}_class_{target_class}.png"

    plt.figure(figsize=(8, 8))
    plt.imshow(base, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=0.35)
    plt.axis("off")
    plt.title(f"Grad-CAM image_id={args.image_id}, class={target_class}, prob={probs[target_class]:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM: {out_path}")


if __name__ == "__main__":
    main()
