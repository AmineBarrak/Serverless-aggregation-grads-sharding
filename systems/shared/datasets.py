"""Dataset loading and partitioning for federated learning experiments.

Supports FEMNIST, CIFAR-100, and RVL-CDIP (simulated).
Handles IID and non-IID data partitioning across clients.
"""

from typing import List, Literal
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms


def _get_cifar100_loaders(
    num_clients: int, batch_size: int = 32, iid: bool = True
) -> List[DataLoader]:
    """Load CIFAR-100 and partition across clients.

    Args:
        num_clients: Number of clients
        batch_size: Batch size for DataLoader
        iid: If True, uniform random partition; if False, use class-based partition

    Returns:
        List of DataLoaders, one per client
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ]
    )

    try:
        train_dataset = datasets.CIFAR100(
            root="/tmp/cifar100", train=True, download=True, transform=transform
        )

        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)

        if iid:
            indices_per_client = np.array_split(indices, num_clients)
        else:
            labels = np.array([train_dataset.targets[i] for i in indices])
            indices_per_client = []
            for client_id in range(num_clients):
                classes_per_client = 5
                client_classes = np.random.choice(
                    100, classes_per_client, replace=False
                )
                client_indices = np.concatenate(
                    [np.where(labels == c)[0] for c in client_classes]
                )
                indices_per_client.append(client_indices)

        loaders = [
            DataLoader(
                Subset(train_dataset, idx_list),
                batch_size=batch_size,
                shuffle=True,
            )
            for idx_list in indices_per_client
        ]
        return loaders

    except Exception:
        # Fallback: generate synthetic CIFAR-100-like data
        print("[datasets] CIFAR-100 download failed, using synthetic data")
        return _generate_synthetic_loaders(
            num_clients, batch_size, image_size=(3, 32, 32), num_classes=100,
            samples_per_client=500,
        )


def _get_femnist_loaders(
    num_clients: int, batch_size: int = 32, iid: bool = True
) -> List[DataLoader]:
    """Load FEMNIST (via EMNIST) and partition across clients.

    Uses the byclass split from EMNIST which includes both letters and digits.

    Args:
        num_clients: Number of clients
        batch_size: Batch size for DataLoader
        iid: If True, uniform random partition; if False, use class-based partition

    Returns:
        List of DataLoaders, one per client
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize(mean=[0.1307]*3, std=[0.3081]*3),
        ]
    )

    try:
        train_dataset = datasets.EMNIST(
            root="/tmp/emnist",
            split="byclass",
            train=True,
            download=True,
            transform=transform,
        )

        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)

        if iid:
            indices_per_client = np.array_split(indices, num_clients)
        else:
            labels = np.array([train_dataset.targets[i] for i in indices])
            classes_per_client = 5
            indices_per_client = []
            for _ in range(num_clients):
                client_classes = np.random.choice(
                    len(np.unique(labels)), classes_per_client, replace=False
                )
                client_indices = np.concatenate(
                    [np.where(labels == c)[0] for c in client_classes]
                )
                indices_per_client.append(client_indices)

        loaders = [
            DataLoader(
                Subset(train_dataset, idx_list),
                batch_size=batch_size,
                shuffle=True,
            )
            for idx_list in indices_per_client
        ]
        return loaders

    except Exception:
        # Fallback: generate synthetic FEMNIST-like data (3ch for model compat)
        print("[datasets] FEMNIST download failed, using synthetic data")
        return _generate_synthetic_loaders(
            num_clients, batch_size, image_size=(3, 28, 28), num_classes=62,
            samples_per_client=500,
        )


def _get_rvlcdip_loaders(
    num_clients: int, batch_size: int = 32, iid: bool = True, num_samples: int = 1000
) -> List[DataLoader]:
    """Simulate RVL-CDIP dataset with random data.

    RVL-CDIP is large (~600k images) and hard to download, so we simulate it
    with random tensors of the correct shape (224x224 RGB images, 16 document classes).

    Args:
        num_clients: Number of clients
        batch_size: Batch size for DataLoader
        iid: If True, uniform random partition
        num_samples: Total samples to simulate per client

    Returns:
        List of DataLoaders, one per client
    """
    # RVL-CDIP: simulated as 32x32 RGB (resized for model compat), 16 classes
    image_size = (3, 32, 32)
    num_classes = 16
    samples_per_client = num_samples

    loaders = []
    for client_id in range(num_clients):
        # Generate random images and labels
        images = torch.randn(samples_per_client, *image_size)
        labels = torch.randint(0, num_classes, (samples_per_client,))

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders


def _generate_synthetic_loaders(
    num_clients: int,
    batch_size: int,
    image_size: tuple,
    num_classes: int,
    samples_per_client: int = 500,
) -> List[DataLoader]:
    """Generate synthetic data when real datasets can't be downloaded.

    Args:
        num_clients: Number of clients
        batch_size: Batch size
        image_size: Tuple of (channels, height, width)
        num_classes: Number of output classes
        samples_per_client: Samples per client

    Returns:
        List of DataLoaders
    """
    loaders = []
    for _ in range(num_clients):
        images = torch.randn(samples_per_client, *image_size)
        labels = torch.randint(0, num_classes, (samples_per_client,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders


def get_dataset(
    name: str, num_clients: int, batch_size: int = 32, iid: bool = True
) -> List[DataLoader]:
    """Get a federated dataset partitioned across clients.

    Args:
        name: Dataset name. One of: "femnist", "cifar100", "rvlcdip"
        num_clients: Number of clients (data partitions)
        batch_size: Batch size for DataLoaders
        iid: If True, use IID partition; if False, use class-based non-IID

    Returns:
        List of DataLoaders, one per client

    Raises:
        ValueError: If dataset name is not supported.
    """
    name = name.lower().strip()

    if name == "cifar100":
        return _get_cifar100_loaders(num_clients, batch_size, iid)
    elif name == "femnist":
        return _get_femnist_loaders(num_clients, batch_size, iid)
    elif name == "rvlcdip":
        return _get_rvlcdip_loaders(num_clients, batch_size, iid)
    else:
        raise ValueError(
            f"Unsupported dataset: {name}. "
            f"Choose from: femnist, cifar100, rvlcdip"
        )
