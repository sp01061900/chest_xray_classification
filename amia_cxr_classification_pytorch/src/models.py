"""Model factory for DenseNet, ResNet, EfficientNet, and ViT classifiers."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
from torchvision import models


def _replace_first_conv(conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    """Adapt a pretrained first convolution to 1-channel or 3-channel input."""
    if conv.in_channels == input_channels:
        return conv

    new_conv = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if input_channels == conv.in_channels else 1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if input_channels == 1 and conv.weight.shape[1] == 3:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        elif input_channels == 3 and conv.weight.shape[1] == 1:
            new_conv.weight.copy_(conv.weight.repeat(1, 3, 1, 1) / 3.0)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout and dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def _load_weights_enum(enum_cls, pretrained: bool):
    if not pretrained:
        return None
    return enum_cls.DEFAULT


def build_xrv_densenet121(num_classes: int, pretrained: bool, weights: str | None, dropout: float) -> nn.Module:
    """Build TorchXRayVision DenseNet121 and replace its classifier head."""
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "torchxrayvision is required for model.name='xrv_densenet121'. "
            "Install requirements.txt or use configs/densenet121_torchvision.yaml."
        ) from exc

    xrv_weights = weights if pretrained else None
    model = xrv.models.DenseNet(weights=xrv_weights)

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = _make_head(in_features, num_classes, dropout)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_linear = None
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError("Could not infer XRV classifier input features")
        model.classifier = _make_head(last_linear.in_features, num_classes, dropout)
    else:
        raise ValueError("Unsupported TorchXRayVision DenseNet classifier structure")
    return model


def build_torchvision_densenet121(num_classes: int, pretrained: bool, input_channels: int, dropout: float) -> nn.Module:
    weights = _load_weights_enum(models.DenseNet121_Weights, pretrained)
    model = models.densenet121(weights=weights)
    model.features.conv0 = _replace_first_conv(model.features.conv0, input_channels)
    in_features = model.classifier.in_features
    model.classifier = _make_head(in_features, num_classes, dropout)
    return model


def build_resnet50(num_classes: int, pretrained: bool, input_channels: int, dropout: float) -> nn.Module:
    weights = _load_weights_enum(models.ResNet50_Weights, pretrained)
    model = models.resnet50(weights=weights)
    model.conv1 = _replace_first_conv(model.conv1, input_channels)
    in_features = model.fc.in_features
    model.fc = _make_head(in_features, num_classes, dropout)
    return model


def build_efficientnet_b3(num_classes: int, pretrained: bool, input_channels: int, dropout: float) -> nn.Module:
    weights = _load_weights_enum(models.EfficientNet_B3_Weights, pretrained)
    model = models.efficientnet_b3(weights=weights)
    first = model.features[0][0]
    model.features[0][0] = _replace_first_conv(first, input_channels)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if dropout is not None:
        model.classifier[0] = nn.Dropout(p=dropout)
    return model


def build_vit_b16(num_classes: int, pretrained: bool, input_channels: int, dropout: float) -> nn.Module:
    weights = _load_weights_enum(models.ViT_B_16_Weights, pretrained)
    model = models.vit_b_16(weights=weights)
    if input_channels != 3:
        model.conv_proj = _replace_first_conv(model.conv_proj, input_channels)
    in_features = model.heads.head.in_features
    model.heads.head = _make_head(in_features, num_classes, dropout)
    return model


def build_model(config: dict) -> nn.Module:
    cfg = config["model"]
    name = cfg["name"].lower()
    num_classes = int(cfg.get("num_classes", 14))
    pretrained = bool(cfg.get("pretrained", True))
    input_channels = int(cfg.get("input_channels", 1))
    dropout = float(cfg.get("dropout", 0.0))

    if name == "xrv_densenet121":
        return build_xrv_densenet121(num_classes, pretrained, cfg.get("weights"), dropout)
    if name == "densenet121":
        return build_torchvision_densenet121(num_classes, pretrained, input_channels, dropout)
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained, input_channels, dropout)
    if name == "efficientnet_b3":
        return build_efficientnet_b3(num_classes, pretrained, input_channels, dropout)
    if name == "vit_b_16":
        return build_vit_b16(num_classes, pretrained, input_channels, dropout)

    raise ValueError(f"Unsupported model name: {cfg['name']}")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except common classifier/head modules."""
    for param in model.parameters():
        param.requires_grad = False

    head_keywords = ["classifier", "fc", "heads"]
    for name, param in model.named_parameters():
        if any(k in name for k in head_keywords):
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_param_groups(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    """Return optimizer parameter groups with a higher LR for classifier heads."""
    head_keywords = ["classifier", "fc", "heads"]
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    return groups
