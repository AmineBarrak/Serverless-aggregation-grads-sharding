"""PyTorch model definitions for federated learning experiments.

Supports standard architectures (ResNet18, EfficientNet-B7, VGG16) and a simple CNN fallback.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN fallback model for testing with 2-5MB parameters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str) -> Tuple[nn.Module, int]:
    """Get a PyTorch model and its parameter count.

    Args:
        name: Model name. One of: "resnet18", "efficientnet_b7", "vgg16", "simple_cnn"

    Returns:
        Tuple of (model, parameter_count)

    Raises:
        ValueError: If model name is not supported.
    """
    name = name.lower().strip()

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify for CIFAR-100 (smaller images)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        # Update final layer for 100 classes (default is 1000)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 100)

    elif name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        # Update final layer for 100 classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 100)

    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Update final layer for 100 classes
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 100)

    elif name == "simple_cnn":
        model = SimpleCNN(num_classes=100)

    else:
        raise ValueError(
            f"Unsupported model: {name}. "
            f"Choose from: resnet18, efficientnet_b7, vgg16, simple_cnn"
        )

    param_count = _count_parameters(model)
    return model, param_count
