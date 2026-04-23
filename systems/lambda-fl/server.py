"""Main server for Lambda-FL federated learning.

Orchestrates the full FL training loop:
1. Initialize model and datasets
2. For each round: distribute model, clients train, collect gradients,
   trigger aggregation, apply updates
3. Collect metrics (latency, cost, memory)
"""

import os
import sys

# Add parent directory to path for shared imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
# Also add own directory for sibling imports
_self_dir = os.path.dirname(os.path.abspath(__file__))
if _self_dir not in sys.path:
    sys.path.insert(0, _self_dir)

from typing import Dict, List, Any, Tuple, Optional
import math
import time
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from shared.config import FLConfig
from shared.models import get_model
from shared.datasets import get_dataset
try:
    from .coordinator import LambdaFLCoordinator
except ImportError:
    from coordinator import LambdaFLCoordinator


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class LambdaFLServer:
    """Main server orchestrating Lambda-FL federated learning."""

    def __init__(self, config: FLConfig):
        """Initialize the FL server.

        Args:
            config: FLConfig with training parameters
        """
        self.config = config
        torch.manual_seed(config.seed)

        # Load model
        self.model, param_count = get_model(config.model_name)
        self.model.eval()
        logger.info(f"Loaded model {config.model_name} with {param_count} parameters")

        # Load datasets for all clients
        self.client_datasets = get_dataset(
            config.dataset_name,
            config.num_clients,
            batch_size=config.batch_size,
        )
        logger.info(f"Loaded {config.dataset_name} for {config.num_clients} clients")

        # Initialize coordinator
        # k = clients per leaf aggregator; default to sqrt(num_clients) for balanced tree
        k = max(2, int(math.ceil(math.sqrt(config.num_clients))))
        self.coordinator = LambdaFLCoordinator(
            num_parties=config.num_clients,
            k=k,
            max_workers=8,
        )

        # Metrics
        self.metrics = {
            "rounds": [],
            "total_time": 0.0,
            "start_time": datetime.now().isoformat(),
        }

    def _get_client_gradients(
        self, client_idx: int, model: nn.Module, epochs: int = 1
    ) -> Dict[str, torch.Tensor]:
        """Train a client locally and return gradients.

        Args:
            client_idx: Client ID
            model: Global model to train
            epochs: Number of local training epochs

        Returns:
            Dictionary of parameter gradients
        """
        device = next(model.parameters()).device

        # Create a copy for local training
        import copy, gc
        local_model = copy.deepcopy(model)
        local_model = local_model.to(device)
        local_model.train()

        # Optimizer
        optimizer = optim.SGD(local_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Get client's data
        dataloader = self.client_datasets[client_idx]

        # Train for specified epochs
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = local_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Compute gradients relative to global model (on CPU)
        grads = {}
        with torch.no_grad():
            for (name, global_param), (_, local_param) in zip(
                model.named_parameters(), local_model.named_parameters()
            ):
                # Gradient = local_param - global_param
                grads[name] = (local_param - global_param).detach().cpu()

        # Clean up local model to free GPU memory
        del local_model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return grads

    def _apply_aggregated_update(
        self, model: nn.Module, aggregated_grads: Dict[str, torch.Tensor]
    ) -> None:
        """Apply aggregated gradients to model (descent step).

        Args:
            model: Model to update
            aggregated_grads: Aggregated gradients from all clients
        """
        with torch.no_grad():
            for (name, param), grad in zip(model.named_parameters(), aggregated_grads.values()):
                param.data = param.data - self.config.learning_rate * grad.to(param.device)

    def run(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Run the full Lambda-FL training loop.

        Returns:
            Tuple of (final_model, metrics)
        """
        logger.info(
            f"Starting Lambda-FL training: {self.config.num_clients} clients, "
            f"{self.config.num_rounds} rounds, k={self.coordinator.k}"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        start_time = time.time()

        # Training loop
        for round_num in range(self.config.num_rounds):
            logger.info(f"===== Round {round_num + 1}/{self.config.num_rounds} =====")
            round_start = time.time()

            # Select clients for this round (use aggregation_goal to sample if specified)
            if self.config.aggregation_goal and self.config.aggregation_goal > 0:
                import random

                selected_clients = random.sample(
                    range(self.config.num_clients),
                    min(self.config.aggregation_goal, self.config.num_clients)
                )
            else:
                selected_clients = list(range(self.config.num_clients))

            # Step 1: Distribute model to clients (simulated)
            logger.info(f"Distributing model to {len(selected_clients)} selected clients")

            # Step 2: Clients train locally and compute gradients
            logger.info("Clients training locally...")
            client_gradients = []

            for client_idx in selected_clients:
                grads = self._get_client_gradients(
                    client_idx, self.model, epochs=self.config.local_epochs
                )
                client_gradients.append(grads)

            logger.info(f"Collected gradients from {len(client_gradients)} clients")

            # Step 3: Trigger aggregation via coordinator
            logger.info("Triggering Lambda-FL aggregation pipeline...")
            try:
                aggregated_grads, agg_metrics = self.coordinator.trigger_aggregation(
                    client_gradients
                )
                logger.info(f"Aggregation completed in {agg_metrics['total_time']:.4f}s")
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")
                raise

            # Step 4: Apply aggregated gradient to model
            logger.info("Applying aggregated update to model")
            self._apply_aggregated_update(self.model, aggregated_grads)

            # Collect round metrics
            round_time = time.time() - round_start
            round_metrics = {
                "round": round_num + 1,
                "num_clients": len(selected_clients),
                "round_time": round_time,
                "aggregation_metrics": agg_metrics,
            }
            self.metrics["rounds"].append(round_metrics)

            logger.info(f"Round {round_num + 1} completed in {round_time:.4f}s")

        total_time = time.time() - start_time
        self.metrics["total_time"] = total_time
        self.metrics["end_time"] = datetime.now().isoformat()

        logger.info(f"Training completed in {total_time:.4f}s")
        logger.info(f"Aggregation metrics: {self.coordinator.get_metrics()}")

        self.coordinator.shutdown()

        return self.model, self.metrics


def run_lambda_fl(config: FLConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    """Convenience function to run Lambda-FL.

    Args:
        config: FLConfig with training parameters

    Returns:
        Tuple of (final_model, metrics)
    """
    server = LambdaFLServer(config)
    return server.run()
