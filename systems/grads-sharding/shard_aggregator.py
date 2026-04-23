"""Shard aggregator: Lambda function that aggregates one shard from all clients."""

import time
from typing import List, Dict, Any, Tuple
import torch
import numpy as np


class ShardAggregator:
    """
    Aggregates a single shard across all clients.
    Each instance represents one Lambda function.
    """

    def __init__(self, shard_id: int, num_clients: int):
        """
        Initialize the shard aggregator.

        Args:
            shard_id: ID of the shard this aggregator handles
            num_clients: Total number of clients
        """
        self.shard_id = shard_id
        self.num_clients = num_clients
        self.execution_times = []
        self.peak_memory_mb = []
        self.cold_start = True
        self.invocations = 0

    def aggregate(self, client_shards: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform FedAvg aggregation on this shard.

        Args:
            client_shards: List of shard tensors from all clients

        Returns:
            Tuple of (aggregated shard, execution metrics)
        """
        start_time = time.time()

        # Core computation: FedAvg
        aggregation_start = time.time()
        aggregated = torch.stack(client_shards).mean(dim=0)
        aggregation_elapsed = time.time() - aggregation_start

        # Simulate memory monitoring
        # Peak memory includes: input gradients + aggregated gradient + intermediate buffers
        total_param_bytes = sum(s.numel() * 4 for s in client_shards)
        avg_input_memory_mb = (total_param_bytes / len(client_shards)) / (1024 * 1024)
        # Stack operation temporarily holds all inputs
        peak_memory_mb = (total_param_bytes / (1024 * 1024)) + avg_input_memory_mb
        self.peak_memory_mb.append(peak_memory_mb)

        total_elapsed = time.time() - start_time
        self.execution_times.append(total_elapsed)
        self.invocations += 1

        metrics = {
            "shard_id": self.shard_id,
            "invocation": self.invocations,
            "total_execution_time_s": total_elapsed,
            "aggregation_compute_time_s": aggregation_elapsed,
            "num_clients": len(client_shards),
            "shard_size_params": aggregated.numel(),
            "peak_memory_mb": peak_memory_mb,
        }

        return aggregated, metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this aggregator's executions."""
        return {
            "shard_id": self.shard_id,
            "total_invocations": self.invocations,
            "avg_execution_time_s": np.mean(self.execution_times) if self.execution_times else 0,
            "max_execution_time_s": max(self.execution_times) if self.execution_times else 0,
            "min_execution_time_s": min(self.execution_times) if self.execution_times else 0,
            "avg_peak_memory_mb": np.mean(self.peak_memory_mb) if self.peak_memory_mb else 0,
            "max_peak_memory_mb": max(self.peak_memory_mb) if self.peak_memory_mb else 0,
            "total_execution_time_s": sum(self.execution_times),
        }


def simulate_lambda_invoke(
    shard_id: int,
    num_clients: int,
    client_shards: List[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Simulate a Lambda function invocation for shard aggregation.

    This is a standalone function that can be called in parallel.

    Args:
        shard_id: ID of the shard
        num_clients: Total number of clients
        client_shards: List of shard tensors from all clients

    Returns:
        Tuple of (aggregated shard, execution metrics)
    """
    aggregator = ShardAggregator(shard_id, num_clients)
    return aggregator.aggregate(client_shards)
