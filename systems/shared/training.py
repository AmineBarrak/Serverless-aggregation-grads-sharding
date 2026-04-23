"""Client-side local training and model evaluation for federated learning.

Provides utilities for local SGD training, gradient application, and model evaluation.
"""

from typing import Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a 1D tensor."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 1,
    lr: float = 0.01,
    device: str = "cpu",
) -> torch.Tensor:
    """Perform local training on a client and return gradient update.

    Computes the gradient update as the difference between final and initial weights.

    Args:
        model: PyTorch model to train
        dataloader: DataLoader for local data
        epochs: Number of local epochs
        lr: Learning rate for SGD
        device: Device to train on ("cpu" or "cuda")

    Returns:
        Gradient update as a flat tensor (on CPU to free GPU memory)
    """
    model.to(device)
    model.train()

    # Store initial weights — keep on same device as model for speed,
    # only the final gradient is moved to CPU
    initial_weights = _flatten_params(model).clone().detach()

    # Setup optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Local training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Compute gradient on device, then move result to CPU to free GPU memory
    final_weights = _flatten_params(model).clone().detach()
    gradient = (final_weights - initial_weights).cpu()

    # Free the weight tensors immediately
    del initial_weights, final_weights

    return gradient


def apply_gradient(model: nn.Module, gradient: torch.Tensor, device: str = "cpu") -> None:
    """Apply an aggregated gradient update to the model.

    Args:
        model: PyTorch model to update
        gradient: Flat gradient tensor to apply
        device: Device model is on
    """
    model.to(device)

    offset = 0
    for param in model.parameters():
        param_size = param.data.numel()
        param.data.add_(gradient[offset : offset + param_size].view_as(param.data))
        offset += param_size


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        Tuple of (accuracy, loss)
    """
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return accuracy, avg_loss
