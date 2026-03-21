"""
Module 2: EfficientNet-B0 Classifier with Grad-CAM.

Fine-tuned CNN for 6-class defect classification with
gradient-weighted class activation mapping for interpretability.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """
    Create EfficientNet-B0 model adapted for grayscale input.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        model: EfficientNet-B0 with modified input and output layers
    """
    # TODO: Load EfficientNet-B0, adapt for grayscale, replace fc layer
    raise NotImplementedError("Task 2.1")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.

    Hooks into the last convolutional layer to compute saliency maps
    showing which regions drove the classification decision.
    """

    def __init__(self, model: nn.Module, target_layer: str = 'features.8'):
        # TODO: Register forward and backward hooks
        raise NotImplementedError("Task 2.4")

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for the input image."""
        raise NotImplementedError("Task 2.4")
