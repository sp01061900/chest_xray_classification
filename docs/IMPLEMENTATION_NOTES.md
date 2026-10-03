# Implementation Notes

## Classification-only interpretation

The challenge source data contains localization columns, but this project converts annotations to image-level classification targets only. Bounding-box coordinates are not used during training, validation, or inference.

## Label handling

`train.csv` can contain multiple rows per image. Each row contributes one class label. The implementation creates a 14-element vector for abnormal findings. Rows with `class_id == 14` are treated as `No finding` and do not set an abnormality label. Therefore:

- Abnormal image: one or more positions in `[0..13]` are `1`.
- Normal image: all 14 positions are `0`.

## NumPy artifacts

The preprocessing and training steps save arrays to make experiments reproducible:

- `train_image_ids.npy`, `val_image_ids.npy`, `test_image_ids.npy`
- `labels_train.npy`, `labels_val.npy`
- `pos_weight.npy`
- `mean.npy`, `std.npy`
- `thresholds.npy`

## Model design

The primary model is `xrv_densenet121`, which attempts to load TorchXRayVision DenseNet121 weights. If TorchXRayVision is unavailable, use the torchvision DenseNet config.
