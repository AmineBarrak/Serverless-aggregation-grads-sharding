"""LIFL Coordinator - the control plane for federated learning.

Manages:
- Worker nodes, gateways, and aggregators
- FL round orchestration (model distribution, update collection, aggregation)
- Placement and autoscaling
- Topology Abstraction Graph (TAG) for aggregator connectivity
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn

try:
    from .shared_memory import SharedMemoryStore
except ImportError:
    from shared_memory import SharedMemoryStore
try:
    from .gateway import NodeGateway
except ImportError:
    from gateway import NodeGateway
try:
    from .aggregator import LIFLAggregator, AggregatorRole
except ImportError:
    from aggregator import LIFLAggregator, AggregatorRole
try:
    from .placement import LocalityAwarePlacement
except ImportError:
    from placement import LocalityAwarePlacement
try:
    from .autoscaler import HierarchyAwareAutoscaler
except ImportError:
    from autoscaler import HierarchyAwareAutoscaler

logger = logging.getLogger(__name__)


@dataclass
class TopologyAbstractionGraph:
    """Topology Abstraction Graph (TAG) describing aggregator connectivity."""
    nodes: Dict[int, List[int]] = field(default_factory=dict)  # node_id -> [aggregator_ids]
    aggregator_parents: Dict[int, Optional[int]] = field(default_factory=dict)  # agg_id -> parent_agg_id
    aggregator_roles: Dict[int, AggregatorRole] = field(default_factory=dict)  # agg_id -> role

    def add_node(self, node_id: int) -> None:
        """Add a node to the TAG."""
        if node_id not in self.nodes:
            self.nodes[node_id] = []

    def add_aggregator(self, aggregator_id: int, node_id: int, role: AggregatorRole, parent: Optional[int] = None) -> None:
        """Add an aggregator to the TAG."""
        if node_id not in self.nodes:
            self.add_node(node_id)
        self.nodes[node_id].append(aggregator_id)
        self.aggregator_parents[aggregator_id] = parent
        self.aggregator_roles[aggregator_id] = role

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TAG to dictionary."""
        return {
            "nodes": self.nodes,
            "aggregator_parents": self.aggregator_parents,
            "aggregator_roles": {k: v.value for k, v in self.aggregator_roles.items()},
        }


class LIFLCoordinator:
    """Coordinator for LIFL federated learning system.

    Orchestrates:
    1. Client placement using LocalityAwarePlacement
    2. Gateway creation for each node
    3. Aggregator pool management via HierarchyAwareAutoscaler
    4. FL round execution with hierarchical aggregation
    """

    def __init__(
        self,
        num_clients: int,
        num_nodes: int = 5,
        max_service_capacity: int = 100,
    ):
        """Initialize the coordinator.

        Args:
            num_clients: Total number of clients
            num_nodes: Number of worker nodes
            max_service_capacity: Max concurrent updates per node
        """
        self.num_clients = num_clients
        self.num_nodes = num_nodes
        self.max_service_capacity = max_service_capacity

        # Initialize shared memory for each node
        self.shared_memories: Dict[int, SharedMemoryStore] = {
            node_id: SharedMemoryStore(max_memory_mb=5000)
            for node_id in range(num_nodes)
        }

        # Create gateways
        self.gateways: Dict[int, NodeGateway] = {}
        clients_per_node = num_clients // num_nodes
        for node_id in range(num_nodes):
            self.gateways[node_id] = NodeGateway(
                node_id=node_id,
                shared_memory=self.shared_memories[node_id],
                num_clients_per_node=clients_per_node,
            )

        # Placement engine
        self.placement = LocalityAwarePlacement(
            num_nodes=num_nodes,
            max_service_capacity=max_service_capacity,
        )

        # Autoscaler
        self.autoscaler = HierarchyAwareAutoscaler(
            shared_memory=self.shared_memories[0],  # Shared memory reference
            num_nodes=num_nodes,
            replan_interval_s=10.0,
        )

        # Client placement
        client_ids = list(range(num_clients))
        self.client_placement = self.placement.place_clients(client_ids)

        # Aggregation state
        self._current_round = 0
        self._lock = threading.RLock()
        self._tag: Optional[TopologyAbstractionGraph] = None
        self._round_metrics: List[Dict[str, Any]] = []

    def distribute_model(self, model: nn.Module) -> Dict[int, torch.Tensor]:
        """Distribute global model to all clients.

        Args:
            model: Global model

        Returns:
            Dictionary mapping client_id -> model weights
        """
        with self._lock:
            # Get model weights
            weights = torch.cat([p.data.flatten() for p in model.parameters()])

            # Distribute to all clients
            client_weights = {}
            for client_id in range(self.num_clients):
                client_weights[client_id] = weights.clone()

            logger.info(f"Distributed model ({weights.numel()} params) to {self.num_clients} clients")
            return client_weights

    def collect_updates(
        self,
        round_num: int,
        client_updates: Dict[int, torch.Tensor],
        timeout_s: float = 30.0,
    ) -> Dict[int, List[str]]:
        """Collect model updates from clients.

        Args:
            round_num: FL round number
            client_updates: Dictionary mapping client_id -> weights
            timeout_s: Timeout for collection

        Returns:
            Dictionary mapping node_id -> list of shared memory keys
        """
        with self._lock:
            self._current_round = round_num
            start_time = time.time()
            collected = {node_id: [] for node_id in range(self.num_nodes)}

            # Send updates to gateways
            for client_id, weights in client_updates.items():
                # Find which node this client is assigned to
                node_id = None
                for nid, client_list in self.client_placement.items():
                    if client_id in client_list:
                        node_id = nid
                        break

                if node_id is None:
                    logger.warning(f"Client {client_id} not placed on any node")
                    continue

                # Send to gateway
                gateway = self.gateways[node_id]
                success = gateway.receive_update(client_id, round_num, weights)

                if success:
                    collected[node_id].extend(
                        gateway.get_pending_updates(round_num)
                    )

            collection_time = time.time() - start_time
            logger.info(f"Collected updates in {collection_time:.2f}s")

            return collected

    def execute_hierarchical_aggregation(
        self,
        round_num: int,
        update_keys_by_node: Dict[int, List[str]],
    ) -> Optional[torch.Tensor]:
        """Execute hierarchical aggregation across nodes.

        Features:
        - Eager aggregation (starts as updates arrive)
        - Locality-aware: aggregates locally first, then upward
        - TAG-based topology management
        - Aggregator reuse

        Args:
            round_num: FL round number
            update_keys_by_node: Updates per node

        Returns:
            Aggregated global model weights or None
        """
        with self._lock:
            agg_start = time.time()

            # Step 1: Plan hierarchy
            num_updates = sum(len(keys) for keys in update_keys_by_node.values())
            hierarchy_plans = self.placement.plan_hierarchy(
                num_updates_per_node=num_updates // self.num_nodes if self.num_nodes > 0 else 1,
                max_k_ary=4,
            )

            # Step 2: Trigger re-planning in autoscaler
            self.autoscaler.trigger_replan(hierarchy_plans)

            # Step 3: Build TAG and execute aggregation per node (concurrently)
            self._tag = TopologyAbstractionGraph()
            final_aggregated_keys: List[str] = []

            def _aggregate_node(node_id: int) -> Optional[str]:
                """Aggregate all updates for a single node. Runs concurrently."""
                update_keys = update_keys_by_node.get(node_id, [])
                if not update_keys:
                    logger.debug(f"Node {node_id}: No updates to aggregate")
                    return None

                plan = hierarchy_plans[node_id]
                logger.debug(
                    f"Node {node_id}: Aggregating {len(update_keys)} updates "
                    f"with k={plan.k_ary}, depth={plan.depth}"
                )

                # Local leaf aggregation
                leaf_results = []
                chunk_size = max(1, len(update_keys) // max(1, plan.leaf_aggregators))
                for i in range(0, len(update_keys), chunk_size):
                    chunk = update_keys[i:i + chunk_size]
                    if not chunk:
                        continue

                    agg = self.autoscaler.get_aggregator(
                        node_id=node_id,
                        role=AggregatorRole.LEAF,
                        reuse=True,
                    )
                    output_key = agg.aggregate(chunk, round_num)
                    if output_key:
                        leaf_results.append(output_key)

                # Middle layer aggregation (if needed)
                if len(leaf_results) > 1:
                    middle_results = []
                    chunk_size = max(1, len(leaf_results) // max(1, plan.middle_aggregators))
                    for i in range(0, len(leaf_results), chunk_size):
                        chunk = leaf_results[i:i + chunk_size]
                        if not chunk:
                            continue

                        agg = self.autoscaler.get_aggregator(
                            node_id=node_id,
                            role=AggregatorRole.MIDDLE,
                            reuse=True,
                        )
                        output_key = agg.aggregate(chunk, round_num)
                        if output_key:
                            middle_results.append(output_key)

                    final_for_node = middle_results
                else:
                    final_for_node = leaf_results

                # Top aggregation (if multiple leaf/middle results)
                if len(final_for_node) > 1:
                    agg = self.autoscaler.get_aggregator(
                        node_id=node_id,
                        role=AggregatorRole.TOP,
                        reuse=True,
                    )
                    output_key = agg.aggregate(final_for_node, round_num)
                    if output_key:
                        return output_key
                elif final_for_node:
                    return final_for_node[0]

                return None

            # Execute per-node aggregation concurrently
            with ThreadPoolExecutor(max_workers=self.num_nodes) as executor:
                futures = {
                    executor.submit(_aggregate_node, node_id): node_id
                    for node_id in range(self.num_nodes)
                }
                for future in as_completed(futures):
                    node_id = futures[future]
                    self._tag.add_node(node_id)
                    try:
                        result_key = future.result()
                        if result_key:
                            final_aggregated_keys.append(result_key)
                    except Exception as e:
                        logger.error(f"Node {node_id} aggregation failed: {e}")
                        raise

                    # Clear processed updates
                    self.gateways[node_id].clear_round_updates(round_num)

            # Step 4: Global aggregation (if multiple nodes)
            if len(final_aggregated_keys) > 1:
                # Use shared memory of first node for global aggregation
                global_agg = self.autoscaler.get_aggregator(
                    node_id=0,
                    role=AggregatorRole.TOP,
                    reuse=True,
                )

                # Copy results to first node's shared memory for aggregation
                for key in final_aggregated_keys:
                    # In real system, would fetch from remote; here we simulate
                    tensor = self.shared_memories[0].get(key)
                    if tensor is None:
                        # Need to fetch from another node (inter-node transfer)
                        # Simulate by getting from first available
                        for node_id in range(1, self.num_nodes):
                            tensor = self.shared_memories[node_id].get(key)
                            if tensor is not None:
                                break

                global_result_key = global_agg.aggregate(
                    final_aggregated_keys, round_num
                )
                final_tensor = self.shared_memories[0].get(global_result_key)
            else:
                final_tensor = self.shared_memories[0].get(final_aggregated_keys[0]) if final_aggregated_keys else None

            agg_time = time.time() - agg_start
            logger.info(f"Hierarchical aggregation completed in {agg_time:.2f}s")

            return final_tensor

    def get_aggregator_topology(self) -> Optional[Dict[str, Any]]:
        """Get the current aggregation topology (TAG).

        Returns:
            Serialized TAG or None
        """
        with self._lock:
            return self._tag.to_dict() if self._tag else None

    def get_coordinator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator statistics.

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            gateway_stats = {
                node_id: gateway.get_statistics()
                for node_id, gateway in self.gateways.items()
            }

            placement_stats = self.placement.get_placement_stats()
            autoscaler_stats = self.autoscaler.get_pool_statistics()

            return {
                "current_round": self._current_round,
                "num_clients": self.num_clients,
                "num_nodes": self.num_nodes,
                "client_placement": self.client_placement,
                "placement_stats": placement_stats,
                "gateway_stats": gateway_stats,
                "autoscaler_stats": autoscaler_stats,
                "topology": self.get_aggregator_topology(),
            }
