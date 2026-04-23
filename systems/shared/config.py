"""Shared configuration for federated learning experiments.

Defines the FLConfig dataclass used across all FL systems.
"""

from dataclasses import dataclass


@dataclass
class FLConfig:
    """Federated learning experiment configuration.

    Attributes:
        num_clients: Number of clients in the federation
        num_rounds: Number of communication rounds
        local_epochs: Number of local training epochs per client per round
        batch_size: Batch size for local training
        learning_rate: SGD learning rate for local training
        model_name: Model architecture (resnet18, efficientnet_b7, vgg16, simple_cnn)
        dataset_name: Dataset (cifar100, femnist, rvlcdip)
        num_shards: Number of shards for gradient sharding (GradsSharding)
        aggregation_goal: Number of clients to wait for before aggregating (0 = all)
        memory_limit_mb: Simulated Lambda memory limit in MB
        timeout_seconds: Timeout for client training in seconds
        seed: Random seed for reproducibility
    """

    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    model_name: str = "resnet18"
    dataset_name: str = "cifar100"
    num_shards: int = 4
    aggregation_goal: int = 0  # 0 means wait for all clients
    memory_limit_mb: int = 512
    timeout_seconds: int = 300
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.aggregation_goal < 0 or self.aggregation_goal > self.num_clients:
            raise ValueError(
                f"aggregation_goal must be between 0 and {self.num_clients}"
            )

        # Model validation
        valid_models = {"resnet18", "efficientnet_b7", "vgg16", "simple_cnn"}
        if self.model_name not in valid_models:
            raise ValueError(
                f"model_name must be one of {valid_models}, got {self.model_name}"
            )

        # Dataset validation
        valid_datasets = {"cifar100", "femnist", "rvlcdip"}
        if self.dataset_name not in valid_datasets:
            raise ValueError(
                f"dataset_name must be one of {valid_datasets}, got {self.dataset_name}"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.__dict__.copy()

    def __str__(self) -> str:
        """String representation for logging."""
        lines = ["FLConfig:"]
        for key, value in self.__dict__.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
