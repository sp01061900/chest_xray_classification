# AMIA Public Challenge 2026 — Chest X-ray Classification Only

This repository contains a **GitHub-ready PyTorch implementation** for chest X-ray **multi-label image classification** based on the AMIA Public Challenge 2026 context. The implementation is intentionally **classification-only**: object detection/localization code is excluded.

## Project scope

- Task: chest X-ray image classification across **14 abnormality classes**.
- Label format: image-level **multi-label** targets created from `train.csv` rows.
- Normal images / `No finding`: represented as an all-zero vector across the 14 abnormality classes.
- Primary model: **DenseNet121** using TorchXRayVision weights when available.
- Baselines: torchvision DenseNet121, ResNet50, EfficientNet-B3, ViT-B/16.
- Loss: `BCEWithLogitsLoss` for multi-label classification.
- Metrics: macro/micro ROC-AUC, PR-AUC, F1, precision, recall, per-class metrics.

## Expected raw data layout

Place the Kaggle/AMIA data under:

```text
data/raw/amia-public-challenge-2026/
├── train.csv
├── sample_submission.csv
├── img_size.csv                 # optional
├── train/
│   └── train/
│       ├── <image_id>.png
│       └── ...
└── test/
    └── test/
        ├── <image_id>.png
        └── ...
```

If your extracted folder layout differs, update `data.raw_dir` in the YAML config files.

## Repository structure

```text
amia-cxr-classification/
├── configs/                     # YAML experiment configs
├── src/                         # source code
├── tests/                       # lightweight unit tests
├── data/                        # raw/processed data, gitignored
├── outputs/                     # checkpoints, metrics, plots, gitignored
├── docs/                        # implementation notes
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: `torchxrayvision` is used for the primary DenseNet121 model. If it is unavailable in your environment, use `configs/densenet121_torchvision.yaml` as a fallback.

## Execution order

### 1. Prepare image-level labels and splits

```bash
python src/prepare_labels.py \
  --raw_dir data/raw/amia-public-challenge-2026 \
  --out_dir data/processed \
  --split_dir data/splits \
  --val_size 0.15 \
  --seed 42
```

Outputs include:

- `data/processed/metadata.csv`
- `data/processed/class_names.json`
- `data/processed/class_counts.csv`
- `data/processed/pos_weight.npy`
- `data/splits/train_image_ids.npy`
- `data/splits/val_image_ids.npy`
- `data/splits/test_image_ids.npy`

### 2. Compute NumPy preprocessing statistics and optional cache

```bash
python src/preprocess_cache.py \
  --raw_dir data/raw/amia-public-challenge-2026 \
  --metadata_csv data/processed/metadata.csv \
  --out_dir data/processed \
  --img_size 224
```

Optional image cache:

```bash
python src/preprocess_cache.py \
  --raw_dir data/raw/amia-public-challenge-2026 \
  --metadata_csv data/processed/metadata.csv \
  --out_dir data/processed \
  --img_size 224 \
  --cache_images
```

### 3. Train primary model

```bash
python src/train.py --config configs/densenet121.yaml
```

### 4. Train comparison models

```bash
python src/train.py --config configs/densenet121_torchvision.yaml
python src/train.py --config configs/resnet50.yaml
python src/train.py --config configs/efficientnet_b3.yaml
python src/train.py --config configs/vit_b16.yaml
```

### 5. Tune thresholds on validation split

```bash
python src/tune_thresholds.py --run_dir outputs/checkpoints/densenet121_xrv
```

### 6. Evaluate and export predictions

```bash
python src/evaluate.py --run_dir outputs/checkpoints/densenet121_xrv --split val
python src/evaluate.py --run_dir outputs/checkpoints/densenet121_xrv --split test
```

For unlabeled test data, `evaluate.py` exports probability predictions instead of metrics.

### 7. Create plots

```bash
python src/plot_results.py \
  --checkpoints_dir outputs/checkpoints \
  --metrics_dir outputs/metrics \
  --out_dir outputs/plots
```

### 8. Optional Grad-CAM for best model

```bash
python src/gradcam.py \
  --run_dir outputs/checkpoints/densenet121_xrv \
  --image_id YOUR_IMAGE_ID \
  --split val \
  --out_dir outputs/plots/gradcam
```

## Key implementation choices

1. **Multi-label target construction**: each image can have multiple abnormality labels. The code aggregates multiple `train.csv` rows per image into a 14-dimensional multi-hot vector.
2. **No finding handling**: `class_id == 14` is treated as normal/all-negative and is not included as a 15th output node.
3. **DenseNet121 primary model**: preferred because DenseNet-style models are widely used for chest X-ray classification, and TorchXRayVision provides chest X-ray pretrained DenseNet weights.
4. **BCEWithLogitsLoss**: appropriate for independent multi-label binary targets and numerically stable because sigmoid and binary cross-entropy are combined internally.
5. **NumPy usage**: label arrays, split arrays, class weights, preprocessing statistics, optional cached images, threshold arrays, and metric aggregation.

## Git notes

Large datasets and generated outputs are intentionally ignored by `.gitignore`. Commit only source files, configs, docs, and tests.

## Run tests

```bash
pytest -q
```
