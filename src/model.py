"""
Module 2: EfficientNet-B0 Classifier with Grad-CAM.

Fine-tuned CNN for 6-class defect classification with
gradient-weighted class activation mapping for interpretability.

Reference: PRD Section 7.2 — EfficientNet-B0 Classifier
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """
    Create EfficientNet-B0 model adapted for grayscale input.

    The pretrained weights expect 3-channel RGB input. For grayscale,
    we average the first conv weights across channels, then replicate
    back to 3 channels (since our preprocessing outputs 3-channel grayscale).

    Args:
        num_classes: Number of output classes (default: 6 for NEU dataset)
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        model: EfficientNet-B0 with modified input and output layers
    """
    # Load pretrained EfficientNet-B0
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # Grayscale adaptation
    # Original first conv: (32, 3, 3, 3)
    # We average RGB weights so the model treats all 3 identical channels equally
    first_conv = model.features[0][0]
    with torch.no_grad():
        # Average across input channels: (32, 3, 3, 3) -> (32, 1, 3, 3)
        avg_weight = first_conv.weight.mean(dim=1, keepdim=True)
        # Replicate to 3 channels: (32, 1, 3, 3) -> (32, 3, 3, 3)
        first_conv.weight = nn.Parameter(avg_weight.repeat(1, 3, 1, 1))

    # Replace classifier head
    # EfficientNet-B0 has 1280 features before final fc
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )

    return model


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.

    Hooks into the last convolutional layer to compute saliency maps
    showing which regions drove the classification decision.

    Reference: "Grad-CAM: Visual Explanations from Deep Networks"
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM.

        Args:
            model: EfficientNet-B0 model
            target_layer: Layer to hook. Default: model.features[-1] (last conv block)
        """
        self.model = model
        self.model.eval()

        # Default: last conv block of EfficientNet-B0
        if target_layer is None:
            target_layer = model.features[-1]

        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor: (1, 3, H, W) input image tensor
            target_class: Class to visualize. If None, uses predicted class.

        Returns:
            heatmap: (H', W') normalized heatmap [0, 1]
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        output[0, target_class].backward()

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU (keep only positive contributions)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: (H, W) or (H, W, 3) original image [0, 255] uint8
            heatmap: (h, w) heatmap from generate() [0, 1]
            alpha: Blend factor (0 = full image, 1 = full heatmap)
            colormap: OpenCV colormap

        Returns:
            overlay: (H, W, 3) BGR image with heatmap overlay
        """
        # Resize heatmap to image size
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )

        # Ensure image is 3-channel BGR
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            image_bgr = image
        else:
            image_bgr = image[:, :, :3]

        # Blend
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay


def load_model(
    checkpoint_path: str,
    num_classes: int = 6,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        num_classes: Number of output classes
        device: Device to load model to

    Returns:
        model: Loaded model in eval mode
    """
    if device is None:
        device = get_device()

    model = create_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> dict:
    """Get model summary statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # float32
    }
