"""Locality-aware placement and hierarchy planning.

Implements:
- BestFit bin-packing for client-to-node assignment
- Dynamic hierarchy planning with EWMA smoothing
- Residual service capacity computation
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)


@dataclass
class NodeCapacity:
    """Represents the capacity state of a node."""
    node_id: int
    max_capacity: int
    current_load: int = 0
    ewma_load: float = 0.0
    clients: List[int] = field(default_factory=list)

    def can_fit(self, client_batch_size: int) -> bool:
        """Check if node can fit additional clients."""
        return self.current_load + client_batch_size <= self.max_capacity

    def add_client(self, client_id: int, batch_size: int = 1) -> None:
        """Add a client to this node."""
        self.current_load += batch_size
        self.clients.append(client_id)

    def get_utilization(self) -> float:
        """Get utilization percentage."""
        if self.max_capacity == 0:
            return 0.0
        return (self.current_load / self.max_capacity) * 100


@dataclass
class HierarchyPlan:
    """Represents an aggregation hierarchy for a node."""
    node_id: int
    k_ary: int  # Number of children per parent
    depth: int  # Depth of the tree
    leaf_aggregators: int
    middle_aggregators: int
    top_aggregator: int
    total_aggregators: int


class LocalityAwarePlacement:
    """Locality-aware client placement and hierarchy planning.

    Features:
    - BestFit bin-packing algorithm
    - EWMA smoothing for load estimation (alpha=0.7)
    - Residual service capacity computation
    - k-ary tree hierarchy generation per node
    """

    EWMA_ALPHA = 0.7  # Smoothing factor from the paper

    def __init__(self, num_nodes: int, max_service_capacity: int):
        """Initialize placement engine.

        Args:
            num_nodes: Number of worker nodes
            max_service_capacity: Maximum concurrent updates per node
        """
        self.num_nodes = num_nodes
        self.max_service_capacity = max_service_capacity
        self.nodes: Dict[int, NodeCapacity] = {
            i: NodeCapacity(node_id=i, max_capacity=max_service_capacity)
            for i in range(num_nodes)
        }
        self._placement_history: List[Dict[int, List[int]]] = []

    def place_clients(self, client_ids: List[int]) -> Dict[int, List[int]]:
        """Assign clients to nodes using BestFit bin-packing.

        Minimizes inter-node communication by fitting clients to the
        most-utilized non-full node (BestFit).

        Args:
            client_ids: List of client IDs to place

        Returns:
            Dictionary mapping node_id -> list of client_ids
        """
        # Reset current loads but keep ewma_load for smoothing
        for node in self.nodes.values():
            node.current_load = 0
            node.clients = []

        placement = {node.node_id: [] for node in self.nodes.values()}

        # BestFit bin-packing: assign each client to node with highest utilization
        # that still has capacity
        for client_id in client_ids:
            best_node = None
            best_utilization = -1

            for node in self.nodes.values():
                if node.can_fit(1):  # Assume batch_size=1 per client
                    util = node.get_utilization()
                    if util > best_utilization:
                        best_utilization = util
                        best_node = node

            if best_node is not None:
                best_node.add_client(client_id, batch_size=1)
                placement[best_node.node_id].append(client_id)
            else:
                # Fallback: assign to node with lowest load
                min_node = min(self.nodes.values(), key=lambda n: n.current_load)
                min_node.add_client(client_id, batch_size=1)
                placement[min_node.node_id].append(client_id)

        self._placement_history.append(placement)
        logger.info(f"Placed {len(client_ids)} clients across {self.num_nodes} nodes")
        for node_id, clients in placement.items():
            if clients:
                logger.debug(f"  Node {node_id}: {len(clients)} clients")

        return placement

    def smooth_load(self, node_id: int, new_load: float) -> float:
        """Apply EWMA smoothing to load estimates.

        Q_smooth = alpha * Q_new + (1-alpha) * Q_old
        Where alpha=0.7 (from paper)

        Args:
            node_id: Node identifier
            new_load: New load measurement

        Returns:
            Smoothed load estimate
        """
        node = self.nodes[node_id]
        node.ewma_load = (
            self.EWMA_ALPHA * new_load +
            (1 - self.EWMA_ALPHA) * node.ewma_load
        )
        return node.ewma_load

    def compute_residual_capacity(
        self,
        node_id: int,
        k: int,
        expected_latency: float
    ) -> float:
        """Compute residual service capacity for a node.

        RC = MC - (k * E)
        Where:
        - MC = max service capacity
        - k = k-ary factor
        - E = expected latency

        Args:
            node_id: Node identifier
            k: Number of children in aggregation tree
            expected_latency: Expected aggregation latency

        Returns:
            Residual capacity (may be negative)
        """
        rc = self.max_service_capacity - (k * expected_latency)
        logger.debug(
            f"Node {node_id}: RC = {self.max_service_capacity} - "
            f"({k} * {expected_latency:.2f}ms) = {rc:.2f}"
        )
        return rc

    def plan_hierarchy(
        self,
        num_updates_per_node: int,
        max_k_ary: int = 8
    ) -> Dict[int, HierarchyPlan]:
        """Plan aggregation hierarchy for all nodes.

        Creates a k-ary tree for each node based on load:
        - More updates -> larger k (more fanout)
        - Fewer updates -> smaller k (less fanout)

        Args:
            num_updates_per_node: Number of updates expected per node
            max_k_ary: Maximum branching factor

        Returns:
            Dictionary mapping node_id -> HierarchyPlan
        """
        plans = {}

        for node_id, node in self.nodes.items():
            # Determine k-ary based on load
            # Load is estimated as number of clients at this node
            if not node.clients:
                # Default hierarchy
                k = 2
                depth = 1
            else:
                num_clients = len(node.clients)

                # Simple heuristic: k = min(max_k_ary, ceil(sqrt(num_clients)))
                # Ensures tree depth stays reasonable
                k = min(max_k_ary, max(2, math.ceil(math.sqrt(num_clients))))
                depth = max(1, math.ceil(math.log(num_clients, k)) if k > 1 else 1)

            # Calculate aggregator counts
            leaf_aggs = math.ceil(num_updates_per_node / k) if k > 0 else 1
            middle_aggs = max(1, math.ceil(leaf_aggs / k)) if k > 1 else 0
            top_agg = 1 if middle_aggs > 0 or leaf_aggs > 1 else 0

            total_aggs = leaf_aggs + middle_aggs + top_agg

            plan = HierarchyPlan(
                node_id=node_id,
                k_ary=k,
                depth=depth,
                leaf_aggregators=leaf_aggs,
                middle_aggregators=middle_aggs,
                top_aggregator=top_agg,
                total_aggregators=total_aggs,
            )

            plans[node_id] = plan
            logger.debug(
                f"Node {node_id}: k={k}, depth={depth}, "
                f"leaves={leaf_aggs}, middle={middle_aggs}, top={top_agg}"
            )

        return plans

    def get_placement_stats(self) -> Dict[str, Any]:
        """Get statistics about current placement.

        Returns:
            Dictionary with placement metrics
        """
        total_clients = sum(len(node.clients) for node in self.nodes.values())
        loaded_nodes = sum(1 for node in self.nodes.values() if node.clients)

        loads = [node.current_load for node in self.nodes.values()]
        avg_load = sum(loads) / len(loads) if loads else 0
        max_load = max(loads) if loads else 0
        min_load = min(loads) if loads else 0

        return {
            "total_clients": total_clients,
            "num_nodes": self.num_nodes,
            "loaded_nodes": loaded_nodes,
            "avg_load_per_node": avg_load,
            "max_load": max_load,
            "min_load": min_load,
            "load_balance": 1.0 - (max(0, max_load - avg_load) / self.max_service_capacity)
            if self.max_service_capacity > 0 else 0.0,
        }

    def get_node_stats(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary with node metrics or None
        """
        if node_id not in self.nodes:
            return None

        node = self.nodes[node_id]
        return {
            "node_id": node_id,
            "current_load": node.current_load,
            "max_capacity": node.max_capacity,
            "utilization_percent": node.get_utilization(),
            "ewma_load": node.ewma_load,
            "num_clients": len(node.clients),
            "client_ids": node.clients.copy(),
        }
