"""
ApertureAI - PCB Surface Defect Inspection System

Modules:
- preprocessing: OpenCV pipeline (M1)
- dataset: PyTorch Dataset class
- augmentations: albumentations pipelines
- model: EfficientNet-B0 classifier (M2a)
- export: ONNX/OpenVINO conversion (M2b)
- optical_simulator: Degradation functions (M3)
- spc: Statistical Process Control (M4)
"""

from .preprocessing import preprocess, preprocess_for_tensor
from .dataset import NEUDataset, get_dataloaders, CLASS_NAMES, NUM_CLASSES
from .augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_heavy_train_transforms,
    get_light_train_transforms,
)

__all__ = [
    # Preprocessing
    "preprocess",
    "preprocess_for_tensor",
    # Dataset
    "NEUDataset",
    "get_dataloaders",
    "CLASS_NAMES",
    "NUM_CLASSES",
    # Augmentations
    "get_train_transforms",
    "get_val_transforms",
    "get_heavy_train_transforms",
    "get_light_train_transforms",
]
