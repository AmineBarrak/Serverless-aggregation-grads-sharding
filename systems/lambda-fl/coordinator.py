"""Coordinator for Lambda-FL tree-based aggregation.

Determines tree topology, triggers leaf and intermediate aggregators,
and manages the message queue-based communication.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from .message_queue import SimulatedKafka
except ImportError:
    from message_queue import SimulatedKafka
try:
    from .aggregator import LeafAggregator, IntermediateAggregator
except ImportError:
    from aggregator import LeafAggregator, IntermediateAggregator
import torch.nn as nn


logger = logging.getLogger(__name__)


def _invoke_leaf_aggregator(
    agg_id: int, k: int, updates: List[Dict[str, any]]
) -> Tuple[Dict[str, any], int, Dict[str, any]]:
    """Invoke a leaf aggregator (simulates Lambda function call).

    Args:
        agg_id: Aggregator ID
        k: Number of updates per aggregator
        updates: List of gradient updates

    Returns:
        Aggregation result (sum_grads, count, metadata)
    """
    agg = LeafAggregator(agg_id, k)
    return agg.aggregate(updates)


def _invoke_intermediate_aggregator(
    agg_id: int, num_leaves: int, partial_results: List[Tuple[Dict[str, any], int]]
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """Invoke an intermediate aggregator (simulates Lambda function call).

    Args:
        agg_id: Aggregator ID
        num_leaves: Number of leaf inputs
        partial_results: List of partial aggregation results

    Returns:
        Final aggregation result (averaged_grads, metadata)
    """
    agg = IntermediateAggregator(agg_id, num_leaves)
    return agg.aggregate(partial_results)


class LambdaFLCoordinator:
    """Orchestrates Lambda-FL federated learning job.

    Manages:
    - Tree topology determination
    - Leaf and intermediate aggregator invocation
    - Message queue-based communication
    - Per-round aggregation orchestration
    """

    def __init__(
        self,
        num_parties: int,
        k: int,
        job_id: str = "job-0",
        max_workers: int = 8,
        timeout: float = 300.0,
    ):
        """Initialize Lambda-FL coordinator.

        Args:
            num_parties: Total number of clients
            k: Number of clients per leaf aggregator
            job_id: Unique job identifier
            max_workers: Maximum concurrent Lambda invocations
            timeout: Timeout for operations (seconds)
        """
        self.num_parties = num_parties
        self.k = k
        self.job_id = job_id
        self.timeout = timeout

        # Message queue for party-aggregator communication
        self.queue = SimulatedKafka()

        # Topic names
        self.parties_topic = f"{job_id}-parties"  # clients publish updates here
        self.agg_topic = f"{job_id}-agg"  # aggregators publish results here

        # Tree topology
        self.num_leaf_aggs = math.ceil(num_parties / k)
        self.aggregators_ready = False
        self._build_tree_topology()

        # Executor for parallel Lambda invocations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Metrics
        self.metrics = {
            "leaf_latencies": [],
            "intermediate_latencies": [],
            "total_invocation_time": 0.0,
            "num_leaf_invocations": 0,
            "num_intermediate_invocations": 0,
        }

        logger.info(
            f"Initialized Lambda-FL coordinator: "
            f"{num_parties} parties, k={k}, "
            f"{self.num_leaf_aggs} leaf aggregators"
        )

    def _build_tree_topology(self) -> None:
        """Build the aggregation tree topology.

        Computes number of leaf aggregators and intermediate aggregators needed.
        """
        # Leaf aggregators: ceil(n/k) aggregators, each handles k clients
        self.leaf_aggs = [
            LeafAggregator(aggregator_id=i, k=self.k) for i in range(self.num_leaf_aggs)
        ]

        # Intermediate aggregators: for now, single-level tree with 1 IA
        # that aggregates all LA outputs
        self.intermediate_aggs = [
            IntermediateAggregator(aggregator_id=0, num_leaves=self.num_leaf_aggs)
        ]

        logger.info(
            f"Tree topology: {self.num_leaf_aggs} leaf aggs -> "
            f"{len(self.intermediate_aggs)} intermediate agg"
        )

    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the aggregation tree.

        Returns:
            Dict with tree structure and configuration
        """
        return {
            "num_parties": self.num_parties,
            "k": self.k,
            "num_leaf_aggregators": self.num_leaf_aggs,
            "num_intermediate_aggregators": len(self.intermediate_aggs),
            "job_id": self.job_id,
        }

    def trigger_aggregation(self, updates: List[Dict[str, any]]) -> Tuple[Dict[str, any], Dict[str, Any]]:
        """Trigger the full aggregation pipeline.

        Orchestrates leaf aggregators, collects their outputs, then runs
        intermediate aggregators.

        Args:
            updates: List of gradient updates from all parties.
                     Length should equal num_parties.

        Returns:
            Tuple of (final_aggregated_model, metrics)
        """
        if len(updates) != self.num_parties:
            raise ValueError(
                f"Expected {self.num_parties} updates but received {len(updates)}"
            )

        round_metrics = {
            "start_time": time.time(),
            "leaf_results": [],
            "intermediate_latencies": [],
        }

        # Stage 1: Invoke leaf aggregators in parallel
        logger.info(f"Invoking {self.num_leaf_aggs} leaf aggregators")
        leaf_futures = []

        for leaf_idx, leaf_agg in enumerate(self.leaf_aggs):
            # Partition updates for this leaf
            start_idx = leaf_idx * self.k
            end_idx = min(start_idx + self.k, len(updates))
            leaf_updates = updates[start_idx:end_idx]

            # Submit leaf aggregator invocation to executor
            future = self.executor.submit(
                _invoke_leaf_aggregator,
                leaf_agg.aggregator_id,
                len(leaf_updates),  # actual count (last leaf may have fewer)
                leaf_updates,
            )
            leaf_futures.append(future)

        # Collect leaf aggregator results
        leaf_results = []
        for i, future in enumerate(as_completed(leaf_futures)):
            try:
                sum_grads, count, metadata = future.result(timeout=self.timeout)
                leaf_results.append((sum_grads, count))
                round_metrics["leaf_results"].append(metadata)
                self.metrics["leaf_latencies"].append(metadata["latency"])
                logger.debug(f"Leaf {i} completed: {metadata}")
            except Exception as e:
                logger.error(f"Leaf aggregator {i} failed: {e}")
                raise

        self.metrics["num_leaf_invocations"] += len(self.leaf_aggs)

        # Stage 2: Invoke intermediate aggregator
        logger.info("Invoking intermediate aggregator")
        intermediate_future = self.executor.submit(
            _invoke_intermediate_aggregator,
            self.intermediate_aggs[0].aggregator_id,
            self.num_leaf_aggs,
            leaf_results,
        )

        try:
            final_grads, ia_metadata = intermediate_future.result(timeout=self.timeout)
            round_metrics["intermediate_latencies"].append(ia_metadata)
            self.metrics["intermediate_latencies"].append(ia_metadata["latency"])
            self.metrics["num_intermediate_invocations"] += 1
            logger.debug(f"Intermediate aggregator completed: {ia_metadata}")
        except Exception as e:
            logger.error(f"Intermediate aggregator failed: {e}")
            raise

        round_metrics["total_time"] = time.time() - round_metrics["start_time"]
        self.metrics["total_invocation_time"] += round_metrics["total_time"]

        return final_grads, round_metrics

    def publish_to_queue(self, topic: str, message: Any) -> None:
        """Publish a message to the message queue.

        Args:
            topic: Topic to publish to
            message: Message payload
        """
        self.queue.publish(topic, message)

    def consume_from_queue(
        self, topic: str, count: int, timeout: Optional[float] = None
    ) -> List[Any]:
        """Consume messages from the message queue.

        Args:
            topic: Topic to consume from
            count: Number of messages to consume
            timeout: Timeout in seconds

        Returns:
            List of messages
        """
        return self.queue.consume(topic, count, timeout=timeout or self.timeout)

    def get_queue_size(self, topic: str) -> int:
        """Get the current size of a topic.

        Args:
            topic: Topic name

        Returns:
            Number of messages in queue
        """
        return self.queue.get_topic_size(topic)

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregation metrics collected so far.

        Returns:
            Dictionary with timing and resource metrics
        """
        return self.metrics.copy()

    def shutdown(self) -> None:
        """Shutdown the coordinator and clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("Lambda-FL coordinator shutdown")
