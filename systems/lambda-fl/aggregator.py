"""Stateless aggregator functions for Lambda-FL.

Implements Leaf Aggregators (LA) and Intermediate Aggregators (IA) as per the paper.
Each aggregator is stateless - all state lives in the message queue.
Simulates ephemeral Lambda function invocations with cold starts.
"""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import time
import logging
import random


logger = logging.getLogger(__name__)


class LeafAggregator:
    """Leaf Aggregator: combines k raw gradient updates into partial aggregate.

    Receives k raw updates from clients, computes sum and count, returns (S_i, k_i).
    Stateless function - simulates a Lambda function invocation.
    """

    def __init__(self, aggregator_id: int, k: int):
        """Initialize a leaf aggregator.

        Args:
            aggregator_id: Unique identifier for this aggregator
            k: Number of client updates to wait for before aggregating
        """
        self.aggregator_id = aggregator_id
        self.k = k
        # Simulate cold start latency (0-50ms)
        self._cold_start_latency = random.uniform(0.001, 0.05)

    def aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], int]:
        """Aggregate k gradient updates from clients.

        Args:
            updates: List of k gradient dictionaries from clients.
                     Each dict maps parameter name to tensor.

        Returns:
            Tuple of (sum_of_gradients, count) representing the partial aggregate.
            sum_of_gradients: Sum of all k updates (not yet averaged)
            count: Number of updates (k)
        """
        start_time = time.time()

        # Simulate cold start
        time.sleep(self._cold_start_latency)

        if len(updates) == 0:
            raise ValueError(
                f"Leaf aggregator {self.aggregator_id} received 0 updates"
            )

        # Sum all gradient updates
        sum_grads = {}
        for param_name in updates[0].keys():
            sum_grads[param_name] = torch.zeros_like(updates[0][param_name])

        for update in updates:
            for param_name, grad in update.items():
                sum_grads[param_name] = sum_grads[param_name] + grad

        latency = time.time() - start_time

        logger.debug(
            f"Leaf aggregator {self.aggregator_id} aggregated {self.k} updates "
            f"in {latency:.4f}s"
        )

        return (sum_grads, self.k, {"latency": latency, "type": "leaf"})

    def __repr__(self) -> str:
        return f"LeafAggregator(id={self.aggregator_id}, k={self.k})"


class IntermediateAggregator:
    """Intermediate Aggregator: combines partial results from leaf aggregators.

    Receives multiple (S_i, k_i) tuples from leaf aggregators, computes final average.
    Result is (1/n) * sum(S_i) where n = sum(k_i).
    Stateless function - simulates a Lambda function invocation.
    """

    def __init__(self, aggregator_id: int, num_leaves: int):
        """Initialize an intermediate aggregator.

        Args:
            aggregator_id: Unique identifier for this aggregator
            num_leaves: Number of leaf aggregator outputs to expect
        """
        self.aggregator_id = aggregator_id
        self.num_leaves = num_leaves
        # Simulate cold start latency (0-50ms)
        self._cold_start_latency = random.uniform(0.001, 0.05)

    def aggregate(
        self, partial_results: List[Tuple[Dict[str, torch.Tensor], int]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Aggregate partial results from leaf aggregators into final average.

        Args:
            partial_results: List of (sum_grads, count) tuples from leaf aggregators.

        Returns:
            Tuple of (averaged_gradients, metadata)
            averaged_gradients: Final aggregated and averaged gradients
            metadata: Timing and statistics
        """
        start_time = time.time()

        # Simulate cold start
        time.sleep(self._cold_start_latency)

        if len(partial_results) == 0:
            raise ValueError(
                f"Intermediate aggregator {self.aggregator_id} received 0 partial results"
            )

        # Aggregate all partial results
        total_count = 0
        aggregated_grads = None

        for sum_grads, count in partial_results:
            total_count += count
            if aggregated_grads is None:
                aggregated_grads = {
                    name: torch.zeros_like(tensor) for name, tensor in sum_grads.items()
                }

            for param_name, grad in sum_grads.items():
                aggregated_grads[param_name] = aggregated_grads[param_name] + grad

        # Average by total count
        for param_name in aggregated_grads:
            aggregated_grads[param_name] = aggregated_grads[param_name] / total_count

        latency = time.time() - start_time

        metadata = {
            "latency": latency,
            "type": "intermediate",
            "total_clients": total_count,
            "num_leaves": self.num_leaves,
        }

        logger.debug(
            f"Intermediate aggregator {self.aggregator_id} aggregated "
            f"{self.num_leaves} leaf results (total {total_count} clients) in {latency:.4f}s"
        )

        return (aggregated_grads, metadata)

    def __repr__(self) -> str:
        return f"IntermediateAggregator(id={self.aggregator_id}, num_leaves={self.num_leaves})"
