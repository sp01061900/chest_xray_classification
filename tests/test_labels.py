from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_labels import build_multilabel_table
from constants import NUM_CLASSES


def test_build_multilabel_table_handles_multiple_classes(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({
        "image_id": ["img1", "img1", "img2"],
        "class_id": [0, 3, 14],
    }).to_csv(csv_path, index=False)

    table = build_multilabel_table(csv_path).sort_values("image_id").reset_index(drop=True)
    label_cols = [f"label_{i}" for i in range(NUM_CLASSES)]

    img1 = table[table["image_id"] == "img1"].iloc[0]
    assert img1["label_0"] == 1
    assert img1["label_3"] == 1
    assert img1[label_cols].sum() == 2

    img2 = table[table["image_id"] == "img2"].iloc[0]
    assert img2[label_cols].sum() == 0


def test_label_vector_has_14_outputs(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"image_id": ["img1"], "class_id": [13]}).to_csv(csv_path, index=False)
    table = build_multilabel_table(csv_path)
    label_cols = [c for c in table.columns if c.startswith("label_")]
    assert len(label_cols) == 14
