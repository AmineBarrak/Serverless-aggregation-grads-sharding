"""LIFL Server - Main entry point for the federated learning system.

Orchestrates:
- Model training on simulated clients
- FL round execution via coordinator
- Metrics collection and reporting
"""

import time
import logging
import sys
import os
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add shared modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import FLConfig
from shared.models import get_model
from shared.datasets import get_dataset
from shared.metrics import MetricsCollector
try:
    from .coordinator import LIFLCoordinator
except ImportError:
    from coordinator import LIFLCoordinator

logger = logging.getLogger(__name__)


class SimulatedClient:
    """Simulates a federated learning client."""

    def __init__(self, client_id: int, model: nn.Module, train_data, test_data, config: FLConfig):
        """Initialize client.

        Args:
            client_id: Client identifier
            model: Neural network model
            train_data: Training dataset
            test_data: Test dataset
            config: FL configuration
        """
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Keep model on CPU until training — avoids OOM when many clients
        # each hold a large model (e.g. VGG-16 × 50 clients)
        self.model = model.cpu()
        self.train_data = train_data
        self.test_data = test_data
        self.config = config

    def train_local_epoch(self) -> Tuple[float, float]:
        """Train for one local epoch.

        Returns:
            Tuple of (loss, accuracy)
        """
        # Move to GPU for training, then back to CPU to free memory
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_data):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_data)
        accuracy = correct / total if total > 0 else 0.0

        # Move model back to CPU to free GPU memory for next client
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, accuracy

    def get_weights(self) -> torch.Tensor:
        """Get current model weights.

        Returns:
            Flattened weight tensor
        """
        return torch.cat([p.data.flatten() for p in self.model.parameters()])

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set model weights.

        Args:
            weights: Flattened weight tensor
        """
        offset = 0
        for param in self.model.parameters():
            numel = param.data.numel()
            param.data = weights[offset:offset + numel].reshape(param.data.shape).to(param.device)
            offset += numel


class LIFLServer:
    """LIFL Federated Learning Server.

    Orchestrates FL training with hierarchical aggregation.
    """

    def __init__(self, config: FLConfig):
        """Initialize server.

        Args:
            config: FL configuration
        """
        self.config = config
        self.metrics = MetricsCollector()

        # Create global model
        self.global_model, self.param_count = get_model(config.model_name)
        logger.info(f"Model: {config.model_name} ({self.param_count} parameters)")

        # Create coordinator
        self.coordinator = LIFLCoordinator(
            num_clients=config.num_clients,
            num_nodes=5,  # Fixed for simulation
            max_service_capacity=config.num_clients,
        )
        logger.info(f"Coordinator initialized with {config.num_clients} clients on {5} nodes")

        # Create simulated clients
        self.clients: Dict[int, SimulatedClient] = {}
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Create and initialize all clients."""
        logger.info(f"Initializing {self.config.num_clients} clients...")

        # Get dataset partitioned across clients
        try:
            # Try to get pre-partitioned datasets
            client_loaders = get_dataset(
                self.config.dataset_name,
                num_clients=self.config.num_clients,
                batch_size=self.config.batch_size,
                iid=True
            )
        except Exception as e:
            logger.warning(f"Failed to load dataset {self.config.dataset_name}: {e}")
            logger.info("Using simulated random data instead")
            client_loaders = self._create_simulated_loaders()

        # Create clients with their data
        for client_id in range(self.config.num_clients):
            # Create model for this client
            model, _ = get_model(self.config.model_name)

            # Get data loader for this client
            client_train_data = client_loaders[client_id] if client_id < len(client_loaders) else None

            # Create test loader (same client's data, no shuffle)
            client_test_data = client_train_data  # Simplified for simulation

            self.clients[client_id] = SimulatedClient(
                client_id=client_id,
                model=model,
                train_data=client_train_data,
                test_data=client_test_data,
                config=self.config,
            )

    def _create_simulated_loaders(self):
        """Create simulated data loaders for testing.

        Returns:
            List of DataLoaders for each client
        """
        loaders = []
        num_samples_per_client = 100

        # Create random tensors matching model input size
        if self.config.dataset_name == "cifar100":
            image_shape = (3, 32, 32)
            num_classes = 100
        elif self.config.dataset_name == "rvlcdip":
            image_shape = (3, 32, 32)
            num_classes = 16
        else:  # femnist
            image_shape = (3, 28, 28)
            num_classes = 62

        for client_id in range(self.config.num_clients):
            images = torch.randn(num_samples_per_client, *image_shape)
            labels = torch.randint(0, num_classes, (num_samples_per_client,))
            dataset = torch.utils.data.TensorDataset(images, labels)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            loaders.append(loader)

        return loaders

    def run(self) -> Tuple[nn.Module, MetricsCollector]:
        """Execute federated learning training.

        Returns:
            Tuple of (final_model, metrics)
        """
        logger.info(f"Starting LIFL training for {self.config.num_rounds} rounds")
        logger.info(f"Config: {self.config}")

        for round_num in range(self.config.num_rounds):
            round_start = time.time()

            # Step 1: Distribute model to clients
            logger.info(f"Round {round_num + 1}/{self.config.num_rounds}")
            client_models = self.coordinator.distribute_model(self.global_model)

            # Step 2: Client training
            client_updates = {}
            for client_id in range(self.config.num_clients):
                # Update client model
                self.clients[client_id].set_weights(client_models[client_id])

                # Train locally
                for _ in range(self.config.local_epochs):
                    loss, acc = self.clients[client_id].train_local_epoch()

                # Get updated weights
                client_updates[client_id] = self.clients[client_id].get_weights()

            # Step 3: Collect updates via gateways (with protocol overhead)
            collect_start = time.time()
            update_keys_by_node = self.coordinator.collect_updates(
                round_num=round_num,
                client_updates=client_updates,
            )
            collection_time = time.time() - collect_start

            # Step 4: Hierarchical aggregation
            agg_start = time.time()
            aggregated_weights = self.coordinator.execute_hierarchical_aggregation(
                round_num=round_num,
                update_keys_by_node=update_keys_by_node,
            )
            agg_time = time.time() - agg_start

            if aggregated_weights is not None:
                # Update global model
                self.global_model = self._set_model_weights(self.global_model, aggregated_weights)
            else:
                logger.warning(f"Round {round_num}: Aggregation failed, keeping previous model")

            round_time = time.time() - round_start

            # Record metrics
            self.metrics.add_round_metric(
                round_num=round_num + 1,
                latency=round_time,
                aggregation_latency=agg_time,
                memory_mb=self.coordinator.shared_memories[0].peak_memory_mb,
            )

            logger.info(
                f"Round {round_num + 1}: "
                f"Round time: {round_time:.2f}s, "
                f"Collection: {collection_time:.2f}s, "
                f"Aggregation: {agg_time:.2f}s"
            )

        logger.info("Training completed!")
        self.metrics.total_lambda_seconds = sum(m["latency_s"] for m in self.metrics.per_round_metrics)
        self.metrics.compute_cost(self.config.memory_limit_mb)

        return self.global_model, self.metrics

    def _set_model_weights(self, model: nn.Module, weights: torch.Tensor) -> nn.Module:
        """Set model weights from flattened tensor.

        Args:
            model: Neural network model
            weights: Flattened weight tensor

        Returns:
            Updated model
        """
        offset = 0
        for param in model.parameters():
            numel = param.data.numel()
            param.data = weights[offset:offset + numel].reshape(param.data.shape).to(param.device)
            offset += numel
        return model

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics.

        Returns:
            Dictionary with metrics
        """
        return {
            "config": self.config.to_dict(),
            "coordinator": self.coordinator.get_coordinator_statistics(),
            "metrics": self.metrics.to_dict(),
        }


def main():
    """Main entry point for LIFL server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create configuration
    config = FLConfig(
        num_clients=10,
        num_rounds=3,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        model_name="simple_cnn",
        dataset_name="cifar100",
        num_shards=4,
    )

    # Create and run server
    server = LIFLServer(config)
    model, metrics = server.run()

    # Print statistics
    logger.info("\n" + metrics.summary())
    logger.info("\nDetailed Statistics:")
    stats = server.get_statistics()
    logger.info(f"Coordinator stats: {stats['coordinator']}")

    return model, metrics


if __name__ == "__main__":
    main()
