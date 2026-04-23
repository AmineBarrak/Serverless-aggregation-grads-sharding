"""GradsSharding server: main FL training loop."""

import os
import sys
import time
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import FLConfig
from shared.models import get_model
from shared.datasets import get_dataset
from shared.training import local_train, evaluate
from shared.metrics import MetricsCollector

try:
    from .orchestrator import ShardOrchestrator
except ImportError:
    from orchestrator import ShardOrchestrator
try:
    from .cost_model import CostModel
except ImportError:
    from cost_model import CostModel


class GradShardingServer:
    """
    GradsSharding federated learning server.

    Implements gradient sharding: the gradient tensor is sharded across
    parallel Lambda aggregators, with each aggregator handling one shard
    from ALL clients (not a subset of clients).
    """

    def __init__(self, config: FLConfig):
        """
        Initialize the server.

        Args:
            config: FL configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model, self.model_size = get_model(config.model_name)
        self.model.to(self.device)

        # Create datasets
        self.train_loaders = get_dataset(
            config.dataset_name,
            config.num_clients,
            config.batch_size,
        )

        # Initialize orchestrator
        self.orchestrator = ShardOrchestrator(config.num_shards)

        # Initialize cost model
        self.cost_model = CostModel()

        # Metrics
        self.metrics_collector = MetricsCollector()

        # Training state
        self.global_model_state = self.model.state_dict()
        self.round_num = 0

    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return self.model_size

    def flatten_model_params(self) -> torch.Tensor:
        """Flatten all model parameters into a single tensor."""
        params = []
        for p in self.model.parameters():
            params.append(p.data.flatten())
        return torch.cat(params)

    def unflatten_to_model(self, flat_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flattened parameters back to state dict, preserving buffers."""
        state_dict = {}
        offset = 0

        for name, param in self.model.named_parameters():
            size = param.numel()
            state_dict[name] = flat_params[offset : offset + size].reshape(param.shape)
            offset += size

        # Preserve BatchNorm buffers (running_mean, running_var, num_batches_tracked)
        # These are not gradients and must not be sharded or averaged
        for name, buffer in self.model.named_buffers():
            state_dict[name] = buffer.clone()

        return state_dict

    def client_train_step(self, client_id: int) -> torch.Tensor:
        """
        Train a client locally and get gradient update.

        Args:
            client_id: Client ID

        Returns:
            Gradient tensor (flattened, on CPU)
        """
        import copy

        # Move global model to CPU temporarily to free GPU for client training
        # This avoids having two full models on GPU simultaneously (OOM for large models)
        self.model.cpu()

        # Create client model directly on GPU
        client_model = copy.deepcopy(self.model)
        client_model.load_state_dict(self.global_model_state)

        # Get client's data loader
        client_loader = self.train_loaders[client_id]

        # Local training — model moves to GPU inside, gradient returned on CPU
        gradient_tensor = local_train(
            client_model,
            client_loader,
            epochs=self.config.local_epochs,
            lr=self.config.learning_rate,
            device=str(self.device),
        )

        # Clean up client model (local_train already moved it off GPU)
        del client_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return gradient_tensor

    def run(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Run federated learning training.

        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        training_start = time.time()

        print(f"\n{'='*80}")
        print(f"GradsSharding Server")
        print(f"{'='*80}")
        print(f"Model: {self.config.model_name}")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Clients: {self.config.num_clients}")
        print(f"Shards: {self.config.num_shards}")
        print(f"Model size: {self.get_model_size():,} parameters")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Local epochs: {self.config.local_epochs}")
        print(f"{'='*80}\n")

        for round_num in range(self.config.num_rounds):
            self.round_num = round_num
            round_start = time.time()

            print(f"Round {round_num + 1}/{self.config.num_rounds}")

            # Sample clients for this round (use all clients)
            num_clients_this_round = self.config.num_clients
            sampled_clients = np.arange(self.config.num_clients)

            # Step 1: Client training
            client_train_start = time.time()
            client_gradients = []

            for client_id in sampled_clients:
                gradient = self.client_train_step(client_id)
                client_gradients.append(gradient)

            # Move global model back to GPU after all client training
            self.model.to(self.device)
            client_train_time = time.time() - client_train_start

            # Step 2: Orchestrate shard aggregation
            aggregation_start = time.time()
            aggregated_gradient, round_metrics = self.orchestrator.orchestrate_round(
                client_gradients,
                self.config.num_shards,
                max_workers=self.config.num_shards,
            )
            aggregation_time = time.time() - aggregation_start

            # Step 3: Apply gradient update to global model
            flat_params = self.flatten_model_params()
            # Ensure gradient is on the same device as model params
            aggregated_gradient = aggregated_gradient.to(flat_params.device)
            updated_params = flat_params - self.config.learning_rate * aggregated_gradient
            new_state_dict = self.unflatten_to_model(updated_params)
            self.model.load_state_dict(new_state_dict)
            self.global_model_state = self.model.state_dict()

            # Step 4: Estimate costs (Lambda compute only — fair comparison)
            max_aggregator_time = round_metrics["max_aggregator_execution_time_s"]
            max_peak_memory = round_metrics["max_peak_memory_mb"]

            cost_breakdown = CostModel.estimate_round_cost(
                num_shards=self.config.num_shards,
                aggregation_time_s=max_aggregator_time,
                memory_mb=max_peak_memory,
            )
            self.cost_model.add_round_cost(cost_breakdown)

            round_elapsed = time.time() - round_start

            # Print round stats
            print(
                f"  Client training: {client_train_time:.2f}s | "
                f"Aggregation: {aggregation_time:.2f}s | "
                f"Total round: {round_elapsed:.2f}s"
            )
            print(
                f"  Max aggregator exec: {max_aggregator_time:.2f}s | "
                f"Peak memory per function: {max_peak_memory:.1f}MB | "
                f"Round cost: ${cost_breakdown['total_cost_usd']:.6f}"
            )

            # Collect metrics
            self.metrics_collector.add_round_metric(
                round_num=round_num,
                latency=round_elapsed,
                aggregation_latency=aggregation_time,
                memory_mb=max_peak_memory,
            )

        training_elapsed = time.time() - training_start

        # Evaluation
        print(f"\n{'='*80}")
        print("Training completed. Evaluating...")
        print(f"{'='*80}\n")

        # Summary metrics
        total_cost = self.cost_model.get_total_cost()
        orchestrator_stats = self.orchestrator.get_stats()

        summary = {
            "training_time_s": training_elapsed,
            "num_rounds": self.config.num_rounds,
            "num_clients": self.config.num_clients,
            "num_shards": self.config.num_shards,
            "model_size": self.get_model_size(),
            "total_cost_usd": total_cost["total_cost_usd"],
            "cost_breakdown": total_cost,
            "orchestrator_stats": orchestrator_stats,
            "metrics": self.metrics_collector.to_dict(),
        }

        print(f"Training completed in {training_elapsed:.2f}s")
        print(f"Total cost (Lambda compute): ${total_cost['total_cost_usd']:.6f}")
        print(f"  Lambda: ${total_cost['lambda_cost_usd']:.6f}")
        print(f"Avg round latency: {orchestrator_stats['avg_round_time_s']:.2f}s")
        avg_mem = np.mean([m.get('peak_memory_mb', 0) for m in self.metrics_collector.per_round_metrics]) if self.metrics_collector.per_round_metrics else 0
        print(f"Avg peak memory per function: {avg_mem:.1f}MB")
        print(f"\n{'='*80}\n")

        return self.model, summary


def main():
    """Main entry point."""
    config = FLConfig(
        num_clients=50,
        num_rounds=5,
        local_epochs=2,
        batch_size=32,
        learning_rate=0.01,
        model_name="resnet18",
        dataset_name="cifar100",
        num_shards=4,
    )

    server = GradShardingServer(config)
    model, summary = server.run()

    return model, summary


if __name__ == "__main__":
    main()
