"""Node gateway for receiving and processing model updates.

Each worker node has one gateway that handles:
1. Protocol processing and deserialization of incoming updates
2. Writing updates to shared memory for local aggregators
3. Routing updates (intra-node via shared memory, inter-node via TCP simulation)
4. Load-based vertical scaling (adjusting CPU cores)
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import torch
try:
    from .shared_memory import SharedMemoryStore
except ImportError:
    from shared_memory import SharedMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Represents a model update from a client."""
    client_id: int
    round_num: int
    weights_key: str  # Shared memory key
    timestamp: float
    size_mb: float


@dataclass
class GatewayStats:
    """Statistics for gateway operations."""
    updates_received: int = 0
    intra_node_transfers: int = 0
    inter_node_transfers: int = 0
    total_bytes_processed: int = 0
    protocol_overhead_ms: float = 0.0
    deserialization_ms: float = 0.0
    cpu_cores_allocated: int = 4


class NodeGateway:
    """Gateway for a worker node handling model update ingestion.

    Features:
    - Protocol processing and deserialization
    - Shared memory integration for zero-copy transfers
    - Intra-node and inter-node routing
    - Vertical scaling based on load
    - Event-driven model for eager aggregation
    """

    def __init__(
        self,
        node_id: int,
        shared_memory: SharedMemoryStore,
        num_clients_per_node: int = 2,
        base_cpu_cores: int = 4,
        max_cpu_cores: int = 16,
    ):
        """Initialize the gateway.

        Args:
            node_id: Identifier for this worker node
            shared_memory: Shared memory store for this node
            num_clients_per_node: Number of clients assigned to this node
            base_cpu_cores: Base CPU cores allocated
            max_cpu_cores: Maximum CPU cores for scaling
        """
        self.node_id = node_id
        self.shared_memory = shared_memory
        self.num_clients_per_node = num_clients_per_node
        self.base_cpu_cores = base_cpu_cores
        self.max_cpu_cores = max_cpu_cores

        self._update_queue: deque[ClientUpdate] = deque()
        self._lock = threading.RLock()
        self._stats = GatewayStats(cpu_cores_allocated=base_cpu_cores)

        # For intra-node aggregators (by round)
        self._aggregator_readiness: Dict[int, List[str]] = {}  # round -> [weight_keys]
        self._processed_updates: Dict[Tuple[int, int], bool] = {}  # (round, client_id) -> processed

    def receive_update(self, client_id: int, round_num: int, weights: torch.Tensor) -> bool:
        """Receive a model update from a client.

        Args:
            client_id: Client identifier
            round_num: FL round number
            weights: Model weights tensor

        Returns:
            True if successfully queued, False otherwise
        """
        with self._lock:
            # Simulate protocol overhead (deserialization, validation)
            process_start = time.time()

            # Protocol processing simulation (protobuf-like overhead)
            protocol_overhead_ms = 1.0  # ms for message processing
            time.sleep(protocol_overhead_ms / 1000)

            # Deserialization simulation (would be real serialization overhead)
            deser_ms = 0.5  # ms for weights deserialization
            time.sleep(deser_ms / 1000)

            # Check if already processed
            if (round_num, client_id) in self._processed_updates:
                logger.warning(f"Duplicate update: round {round_num}, client {client_id}")
                return False

            # Write to shared memory
            weights_key = f"update_r{round_num}_c{client_id}_n{self.node_id}"
            size_bytes = weights.element_size() * weights.nelement()
            size_mb = size_bytes / (1024 * 1024)

            try:
                self.shared_memory.put(weights_key, weights)
            except RuntimeError as e:
                logger.error(f"Failed to store update in shared memory: {e}")
                return False

            # Create update record
            update = ClientUpdate(
                client_id=client_id,
                round_num=round_num,
                weights_key=weights_key,
                timestamp=time.time(),
                size_mb=size_mb,
            )

            self._update_queue.append(update)
            self._processed_updates[(round_num, client_id)] = True

            # Track statistics
            self._stats.updates_received += 1
            self._stats.total_bytes_processed += size_bytes
            self._stats.protocol_overhead_ms += protocol_overhead_ms
            self._stats.deserialization_ms += deser_ms
            self._stats.intra_node_transfers += 1

            # Track readiness for aggregators
            if round_num not in self._aggregator_readiness:
                self._aggregator_readiness[round_num] = []
            self._aggregator_readiness[round_num].append(weights_key)

            logger.debug(
                f"Node {self.node_id}: Received update from client {client_id} "
                f"(round {round_num}, {size_mb:.2f}MB)"
            )

            return True

    def get_pending_updates(self, round_num: int) -> List[str]:
        """Get pending weight keys for a round (for aggregators).

        Args:
            round_num: FL round number

        Returns:
            List of shared memory keys
        """
        with self._lock:
            return self._aggregator_readiness.get(round_num, []).copy()

    def clear_round_updates(self, round_num: int) -> None:
        """Clear updates for a completed round.

        Args:
            round_num: FL round number
        """
        with self._lock:
            if round_num in self._aggregator_readiness:
                self._aggregator_readiness.pop(round_num)

    def route_to_local_aggregator(self, weights_key: str) -> str:
        """Route an update to a local aggregator via shared memory.

        Args:
            weights_key: Shared memory key for weights

        Returns:
            The same key (intra-node routing, no transformation)
        """
        # In this implementation, we use shared memory keys for intra-node routing
        return weights_key

    def route_to_remote_node(self, weights_key: str, target_node_id: int) -> Optional[str]:
        """Simulate routing to a remote node (inter-node transfer).

        Args:
            weights_key: Shared memory key
            target_node_id: Target node ID

        Returns:
            Remote key or None if failed
        """
        with self._lock:
            # Simulate network overhead
            network_latency_ms = 5.0  # Base latency
            serialization_ms = 2.0  # Serialization overhead
            time.sleep((network_latency_ms + serialization_ms) / 1000)

            self._stats.inter_node_transfers += 1

            remote_key = f"{weights_key}->n{target_node_id}"
            logger.debug(f"Routing {weights_key} to node {target_node_id}")
            return remote_key

    def scale_cpu_cores(self, load: float) -> int:
        """Adjust CPU cores based on load (vertical scaling).

        Args:
            load: Load metric (0.0 to 1.0+)

        Returns:
            New number of allocated cores
        """
        with self._lock:
            # EWMA-based scaling with alpha=0.7
            alpha = 0.7
            # Current load is ewma_load = alpha * new + (1-alpha) * old
            if load > 0.8:
                # Scale up
                new_cores = min(
                    int(self._stats.cpu_cores_allocated * 1.5),
                    self.max_cpu_cores
                )
            elif load < 0.3:
                # Scale down
                new_cores = max(
                    int(self._stats.cpu_cores_allocated * 0.67),
                    self.base_cpu_cores
                )
            else:
                new_cores = self._stats.cpu_cores_allocated

            self._stats.cpu_cores_allocated = new_cores
            logger.debug(f"Node {self.node_id}: Scaled to {new_cores} cores (load={load:.2f})")
            return new_cores

    def get_queue_length(self) -> int:
        """Get number of pending updates."""
        with self._lock:
            return len(self._update_queue)

    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics.

        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            return {
                "node_id": self.node_id,
                "updates_received": self._stats.updates_received,
                "intra_node_transfers": self._stats.intra_node_transfers,
                "inter_node_transfers": self._stats.inter_node_transfers,
                "total_bytes_processed": self._stats.total_bytes_processed,
                "protocol_overhead_ms": self._stats.protocol_overhead_ms,
                "deserialization_ms": self._stats.deserialization_ms,
                "cpu_cores_allocated": self._stats.cpu_cores_allocated,
                "pending_updates": len(self._update_queue),
            }
