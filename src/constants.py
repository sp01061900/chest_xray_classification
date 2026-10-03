"""Project constants for AMIA chest X-ray classification."""

NUM_CLASSES = 14
NO_FINDING_CLASS_ID = 14

# AMIA/VinBigData abnormality classes used for classification.
# The order is class_id 0..13. "No finding" is class_id 14 and is represented
# by an all-zero abnormality vector, not by an extra model output.
CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
