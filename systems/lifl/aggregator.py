"""Stateless aggregators for hierarchical model aggregation.

Features:
- Step-based processing: Recv -> Agg -> Send
- Eager aggregation (starts as soon as updates arrive)
- Can serve as leaf, middle, or top aggregator
- Tracks execution metrics (simulated eBPF sidecar)
- Reusable for warm starts
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn.functional as F
try:
    from .shared_memory import SharedMemoryStore
except ImportError:
    from shared_memory import SharedMemoryStore

logger = logging.getLogger(__name__)


class AggregatorRole(Enum):
    """Role in the aggregation hierarchy."""
    LEAF = "leaf"
    MIDDLE = "middle"
    TOP = "top"


@dataclass
class AggregationMetrics:
    """Metrics for an aggregation task."""
    aggregator_id: int
    round_num: int
    role: AggregatorRole
    num_inputs: int = 0
    recv_time_ms: float = 0.0
    agg_time_ms: float = 0.0
    send_time_ms: float = 0.0
    total_time_ms: float = 0.0
    output_key: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aggregator_id": self.aggregator_id,
            "round_num": self.round_num,
            "role": self.role.value,
            "num_inputs": self.num_inputs,
            "recv_time_ms": self.recv_time_ms,
            "agg_time_ms": self.agg_time_ms,
            "send_time_ms": self.send_time_ms,
            "total_time_ms": self.total_time_ms,
            "output_key": self.output_key,
        }


class LIFLAggregator:
    """Stateless aggregator for hierarchical model aggregation.

    Processes gradients in three steps:
    1. Recv: Dequeue updates from shared memory
    2. Agg: Average the updates
    3. Send: Write result to shared memory

    Can be reused across rounds/roles (warm starts).
    """

    # Class-level counter for aggregator IDs
    _id_counter = 0
    _counter_lock = threading.Lock()

    @classmethod
    def _get_next_id(cls) -> int:
        """Get next aggregator ID."""
        with cls._counter_lock:
            cls._id_counter += 1
            return cls._id_counter

    def __init__(self, shared_memory: SharedMemoryStore):
        """Initialize a stateless aggregator.

        Args:
            shared_memory: Reference to shared memory for I/O
        """
        self.aggregator_id = self._get_next_id()
        self.shared_memory = shared_memory
        self._role = AggregatorRole.LEAF
        self._lock = threading.RLock()
        self._metrics_history: List[AggregationMetrics] = []
        self._total_aggregations = 0
        self._warm_start = False

    def set_role(self, role: AggregatorRole) -> None:
        """Set the aggregator's role in the hierarchy.

        Args:
            role: New role (leaf, middle, or top)
        """
        with self._lock:
            self._role = role

    def get_role(self) -> AggregatorRole:
        """Get current role."""
        with self._lock:
            return self._role

    def mark_warm_start(self) -> None:
        """Mark this aggregator as starting from a warm state (reuse)."""
        with self._lock:
            self._warm_start = True

    def is_warm_start(self) -> bool:
        """Check if this is a warm start."""
        with self._lock:
            return self._warm_start

    def aggregate(
        self,
        input_keys: List[str],
        round_num: int,
        output_key: Optional[str] = None,
    ) -> Optional[str]:
        """Aggregate model updates from shared memory.

        Implements the three-step pipeline:
        1. Recv: Load tensors from shared memory
        2. Agg: Average them
        3. Send: Write result back to shared memory

        Args:
            input_keys: List of shared memory keys to aggregate
            round_num: FL round number
            output_key: Optional custom output key

        Returns:
            Output key in shared memory, or None if failed
        """
        with self._lock:
            metrics = AggregationMetrics(
                aggregator_id=self.aggregator_id,
                round_num=round_num,
                role=self._role,
                num_inputs=len(input_keys),
            )

            if not input_keys:
                logger.warning(f"Aggregator {self.aggregator_id}: No inputs to aggregate")
                return None

            try:
                # STEP 1: RECV - Dequeue from shared memory (with timing)
                recv_start = time.time()
                tensors = []
                for key in input_keys:
                    tensor = self.shared_memory.get(key)
                    if tensor is None:
                        logger.warning(f"Aggregator {self.aggregator_id}: Key {key} not found")
                        continue
                    tensors.append(tensor)
                recv_time_ms = (time.time() - recv_start) * 1000

                if not tensors:
                    logger.error(f"Aggregator {self.aggregator_id}: Failed to retrieve any tensors")
                    return None

                # STEP 2: AGG - Aggregate (average) the updates
                agg_start = time.time()
                aggregated = self._aggregate_tensors(tensors)
                agg_time_ms = (time.time() - agg_start) * 1000

                # STEP 3: SEND - Write result to shared memory
                send_start = time.time()
                if output_key is None:
                    output_key = (
                        f"agg_r{round_num}_a{self.aggregator_id}_"
                        f"{self._role.value}"
                    )

                self.shared_memory.put(output_key, aggregated)
                send_time_ms = (time.time() - send_start) * 1000

                # Record metrics
                metrics.recv_time_ms = recv_time_ms
                metrics.agg_time_ms = agg_time_ms
                metrics.send_time_ms = send_time_ms
                metrics.total_time_ms = recv_time_ms + agg_time_ms + send_time_ms
                metrics.output_key = output_key
                metrics.completed_at = time.time()

                self._metrics_history.append(metrics)
                self._total_aggregations += 1

                logger.debug(
                    f"Aggregator {self.aggregator_id} ({self._role.value}): "
                    f"Aggregated {len(tensors)} inputs in {metrics.total_time_ms:.2f}ms"
                )

                return output_key

            except Exception as e:
                logger.error(f"Aggregator {self.aggregator_id}: Aggregation failed: {e}")
                return None

    def _aggregate_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Average a list of tensors.

        Args:
            tensors: List of tensors to aggregate

        Returns:
            Averaged tensor
        """
        # Stack and average
        stacked = torch.stack(tensors, dim=0)
        aggregated = torch.mean(stacked, dim=0)
        return aggregated

    def eager_aggregate_stream(
        self,
        input_keys_stream: List[str],
        round_num: int,
        min_inputs: int = 1,
        output_key: Optional[str] = None,
    ) -> Optional[str]:
        """Perform eager aggregation as updates arrive.

        This simulates the eager aggregation feature where aggregation
        can start before all inputs are available, overlapping with
        network transfer latency.

        Args:
            input_keys_stream: List of keys arriving over time
            round_num: FL round number
            min_inputs: Minimum inputs before starting aggregation
            output_key: Optional custom output key

        Returns:
            Output key of the final aggregated result
        """
        with self._lock:
            # For simulation, eager aggregation is represented by
            # starting to aggregate as soon as min_inputs are available
            if len(input_keys_stream) >= min_inputs:
                return self.aggregate(
                    input_keys_stream[:min_inputs],
                    round_num,
                    output_key
                )
            return None

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get aggregation metrics history.

        Returns:
            List of metric dictionaries
        """
        with self._lock:
            return [m.to_dict() for m in self._metrics_history]

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of this aggregator's executions.

        Returns:
            Summary statistics
        """
        with self._lock:
            if not self._metrics_history:
                return {
                    "aggregator_id": self.aggregator_id,
                    "total_aggregations": 0,
                    "avg_time_ms": 0.0,
                    "total_inputs": 0,
                }

            total_time = sum(m.total_time_ms for m in self._metrics_history)
            total_inputs = sum(m.num_inputs for m in self._metrics_history)
            avg_time = total_time / len(self._metrics_history)

            return {
                "aggregator_id": self.aggregator_id,
                "total_aggregations": self._total_aggregations,
                "avg_time_ms": avg_time,
                "min_time_ms": min(m.total_time_ms for m in self._metrics_history),
                "max_time_ms": max(m.total_time_ms for m in self._metrics_history),
                "total_inputs": total_inputs,
                "warm_start": self._warm_start,
            }
