"""Orchestration layer: coordinates shard aggregation using AWS Step Functions."""

import time
from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .shard_manager import ShardManager, ShardStore
    from .shard_aggregator import simulate_lambda_invoke
except ImportError:
    from shard_manager import ShardManager, ShardStore
    from shard_aggregator import simulate_lambda_invoke


class ShardOrchestrator:
    """
    Orchestrates gradient aggregation across parallel shard aggregators.
    Simulates AWS Step Functions for coordination.
    """

    def __init__(self, num_shards: int):
        """
        Initialize the orchestrator.

        Args:
            num_shards: Number of shards to aggregate in parallel
        """
        self.num_shards = num_shards
        self.shard_manager = ShardManager()
        self.shard_store = ShardStore()
        self.round_times = []

    def orchestrate_round(
        self,
        client_gradients: List[torch.Tensor],
        num_shards: int,
        max_workers: int = 4,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Orchestrate a complete aggregation round.

        Steps:
        1. Each client splits gradient and uploads shards to ShardStore
        2. Launch parallel shard aggregators (one per shard)
        3. Each aggregator downloads its shard from all clients and aggregates
        4. Merge all aggregated shards into full gradient

        Args:
            client_gradients: List of gradient tensors from all clients (one per client)
            num_shards: Number of shards
            max_workers: Max parallel workers

        Returns:
            Tuple of (aggregated gradient, round metrics)
        """
        round_start = time.time()
        round_num = len(self.round_times)
        num_clients = len(client_gradients)

        # Step 1: Client sharding and uploading
        shard_upload_start = time.time()

        for client_id, gradient in enumerate(client_gradients):
            shards = self.shard_manager.split_gradient(gradient, num_shards)
            for shard_id, shard in enumerate(shards):
                self.shard_store.upload_shard(client_id, shard_id, round_num, shard)

        shard_upload_time = time.time() - shard_upload_start

        # Step 2: Launch parallel shard aggregators
        aggregation_start = time.time()

        aggregated_shards = [None] * num_shards
        aggregation_metrics = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for shard_id in range(num_shards):
                # Download this shard from all clients
                client_shards, download_time = self.shard_store.download_shards(
                    shard_id, round_num, list(range(num_clients))
                )

                # Submit aggregation task
                future = executor.submit(
                    simulate_lambda_invoke,
                    shard_id,
                    num_clients,
                    client_shards,
                )
                futures[future] = shard_id

            # Collect results as they complete
            for future in as_completed(futures):
                shard_id = futures[future]
                aggregated_shard, metrics = future.result()
                aggregated_shards[shard_id] = aggregated_shard
                aggregation_metrics.append(metrics)

        aggregation_time = time.time() - aggregation_start

        # Step 3: Merge shards
        merge_start = time.time()

        aggregated_gradient = self.shard_manager.merge_shards(aggregated_shards)

        merge_time = time.time() - merge_start

        # Step 4: Cleanup
        cleanup_start = time.time()
        self.shard_store.clear_round(round_num)

        cleanup_time = time.time() - cleanup_start
        round_elapsed = time.time() - round_start

        self.round_times.append(round_elapsed)

        # Compute summary metrics
        execution_times = [m["total_execution_time_s"] for m in aggregation_metrics]
        peak_memories = [m["peak_memory_mb"] for m in aggregation_metrics]

        round_metrics = {
            "round_num": round_num,
            "total_round_time_s": round_elapsed,
            "client_shard_upload_time_s": shard_upload_time,
            "parallel_aggregation_time_s": aggregation_time,
            "max_aggregator_execution_time_s": max(execution_times) if execution_times else 0,
            "min_aggregator_execution_time_s": min(execution_times) if execution_times else 0,
            "avg_aggregator_execution_time_s": np.mean(execution_times) if execution_times else 0,
            "merge_time_s": merge_time,
            "cleanup_time_s": cleanup_time,
            "num_clients": num_clients,
            "num_shards": num_shards,
            "num_parallel_aggregators": num_shards,
            "max_peak_memory_mb": max(peak_memories) if peak_memories else 0,
            "avg_peak_memory_mb": np.mean(peak_memories) if peak_memories else 0,
            "aggregation_metrics": aggregation_metrics,
        }

        return aggregated_gradient, round_metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get overall orchestration statistics."""
        return {
            "total_rounds": len(self.round_times),
            "avg_round_time_s": np.mean(self.round_times) if self.round_times else 0,
            "max_round_time_s": max(self.round_times) if self.round_times else 0,
            "min_round_time_s": min(self.round_times) if self.round_times else 0,
            "total_round_time_s": sum(self.round_times),
            "shard_manager_stats": self.shard_manager.get_stats(),
            "shard_store_stats": self.shard_store.get_stats(),
        }
