"""
PyTorch Dataset class for NEU Surface Defect Database.

Handles stratified train/val/test splits and augmentation pipeline.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class NEUDataset(Dataset):
    """
    NEU Surface Defect Dataset.

    Args:
        root: Path to dataset root directory
        split: One of 'train', 'val', 'test'
        transform: albumentations transform pipeline
    """

    def __init__(self, root: str, split: str = 'train', transform=None):
        # TODO: Implement dataset loading with stratified split
        raise NotImplementedError("Task 1.3")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
