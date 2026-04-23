"""Autoscaler for dynamically managing aggregators.

Features:
- Periodic re-planning of hierarchy based on load
- Aggregator reuse and promotion
- Warm starts for idle aggregators
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
try:
    from .aggregator import LIFLAggregator, AggregatorRole
except ImportError:
    from aggregator import LIFLAggregator, AggregatorRole
try:
    from .shared_memory import SharedMemoryStore
except ImportError:
    from shared_memory import SharedMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class AggregatorInstance:
    """Represents an active aggregator instance."""
    aggregator: LIFLAggregator
    role: AggregatorRole
    node_id: int
    created_at: float
    last_used_at: float
    completed_aggregations: int = 0

    def is_idle(self, idle_threshold_s: float = 5.0) -> bool:
        """Check if this aggregator is idle."""
        return (time.time() - self.last_used_at) > idle_threshold_s

    def get_age_s(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at


class HierarchyAwareAutoscaler:
    """Manages aggregator pool with reuse and warm starts.

    Features:
    - Creates/terminates aggregators on demand
    - Promotes idle leaf aggregators to middle/top roles
    - Tracks aggregator reuse statistics
    - Periodically re-plans hierarchy
    """

    def __init__(
        self,
        shared_memory: SharedMemoryStore,
        num_nodes: int,
        replan_interval_s: float = 10.0,
    ):
        """Initialize autoscaler.

        Args:
            shared_memory: Reference to shared memory
            num_nodes: Number of worker nodes
            replan_interval_s: How often to re-plan hierarchy (seconds)
        """
        self.shared_memory = shared_memory
        self.num_nodes = num_nodes
        self.replan_interval_s = replan_interval_s

        # Pool of aggregators by node and role
        self._pool: Dict[int, Dict[AggregatorRole, List[AggregatorInstance]]] = {
            node_id: {
                AggregatorRole.LEAF: [],
                AggregatorRole.MIDDLE: [],
                AggregatorRole.TOP: [],
            }
            for node_id in range(num_nodes)
        }

        self._lock = threading.RLock()
        self._last_replan = time.time()
        self._replan_count = 0
        self._aggregator_reuse_count = 0
        self._aggregator_creation_count = 0
        self._aggregator_termination_count = 0

    def get_aggregator(
        self,
        node_id: int,
        role: AggregatorRole,
        reuse: bool = True,
    ) -> LIFLAggregator:
        """Get or create an aggregator for a role.

        Implements aggregator reuse: tries to reuse an idle aggregator
        before creating a new one. Idle leaf aggregators can be promoted
        to higher roles.

        Args:
            node_id: Node ID where aggregator will run
            role: Desired role (leaf, middle, top)
            reuse: Whether to attempt reuse

        Returns:
            An aggregator instance
        """
        with self._lock:
            pool = self._pool[node_id][role]

            # Try to find an idle aggregator to reuse
            if reuse:
                for idx, instance in enumerate(pool):
                    if instance.is_idle():
                        # Reuse this aggregator
                        instance.last_used_at = time.time()
                        instance.completed_aggregations += 1
                        instance.aggregator.mark_warm_start()
                        self._aggregator_reuse_count += 1

                        logger.debug(
                            f"Reusing aggregator {instance.aggregator.aggregator_id} "
                            f"(role={role.value}, age={instance.get_age_s():.1f}s)"
                        )
                        return instance.aggregator

            # Also try to promote an idle leaf aggregator to higher roles
            if role != AggregatorRole.LEAF and reuse:
                leaf_pool = self._pool[node_id][AggregatorRole.LEAF]
                for idx, instance in enumerate(leaf_pool):
                    if instance.is_idle():
                        # Promote to requested role
                        old_role = instance.role
                        instance.role = role
                        instance.aggregator.set_role(role)
                        instance.last_used_at = time.time()
                        instance.aggregator.mark_warm_start()

                        # Move to new pool
                        leaf_pool.pop(idx)
                        self._pool[node_id][role].append(instance)
                        self._aggregator_reuse_count += 1

                        logger.debug(
                            f"Promoted aggregator {instance.aggregator.aggregator_id} "
                            f"from {old_role.value} to {role.value}"
                        )
                        return instance.aggregator

            # Create a new aggregator
            agg = LIFLAggregator(self.shared_memory)
            agg.set_role(role)

            instance = AggregatorInstance(
                aggregator=agg,
                role=role,
                node_id=node_id,
                created_at=time.time(),
                last_used_at=time.time(),
                completed_aggregations=0,
            )

            pool.append(instance)
            self._aggregator_creation_count += 1

            logger.debug(
                f"Created aggregator {agg.aggregator_id} "
                f"(node={node_id}, role={role.value})"
            )
            return agg

    def terminate_aggregator(self, aggregator_id: int) -> bool:
        """Explicitly terminate an aggregator.

        Args:
            aggregator_id: ID of aggregator to terminate

        Returns:
            True if terminated, False if not found
        """
        with self._lock:
            for node_id in range(self.num_nodes):
                for role in AggregatorRole:
                    pool = self._pool[node_id][role]
                    for idx, instance in enumerate(pool):
                        if instance.aggregator.aggregator_id == aggregator_id:
                            pool.pop(idx)
                            self._aggregator_termination_count += 1
                            logger.debug(
                                f"Terminated aggregator {aggregator_id}"
                            )
                            return True
            return False

    def trigger_replan(self, hierarchy_plans: Dict[int, Any]) -> None:
        """Trigger re-planning of the aggregator hierarchy.

        Can terminate/create aggregators based on new plan.

        Args:
            hierarchy_plans: Dictionary of node_id -> HierarchyPlan
        """
        with self._lock:
            current_time = time.time()
            if current_time - self._last_replan < self.replan_interval_s:
                return  # Too soon for re-plan

            self._replan_count += 1
            logger.info(f"Hierarchy re-plan #{self._replan_count}")

            # For each node, check if we need to adjust aggregator count
            for node_id, plan in hierarchy_plans.items():
                current_leaf = len(self._pool[node_id][AggregatorRole.LEAF])
                current_middle = len(self._pool[node_id][AggregatorRole.MIDDLE])
                current_top = len(self._pool[node_id][AggregatorRole.TOP])

                # Terminate excess aggregators
                if current_leaf > plan.leaf_aggregators:
                    excess = current_leaf - plan.leaf_aggregators
                    logger.debug(
                        f"Node {node_id}: Terminating {excess} excess leaf aggregators"
                    )
                    # Terminate idle ones first
                    pool = self._pool[node_id][AggregatorRole.LEAF]
                    removed = 0
                    for idx in range(len(pool) - 1, -1, -1):
                        if removed >= excess:
                            break
                        if pool[idx].is_idle():
                            pool.pop(idx)
                            self._aggregator_termination_count += 1
                            removed += 1

                if current_middle > plan.middle_aggregators:
                    excess = current_middle - plan.middle_aggregators
                    logger.debug(
                        f"Node {node_id}: Terminating {excess} excess middle aggregators"
                    )
                    pool = self._pool[node_id][AggregatorRole.MIDDLE]
                    removed = 0
                    for idx in range(len(pool) - 1, -1, -1):
                        if removed >= excess:
                            break
                        if pool[idx].is_idle():
                            pool.pop(idx)
                            self._aggregator_termination_count += 1
                            removed += 1

            self._last_replan = current_time

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics about the aggregator pool.

        Returns:
            Dictionary with pool metrics
        """
        with self._lock:
            total_aggregators = 0
            by_role = {role: 0 for role in AggregatorRole}
            idle_count = 0

            for node_id in range(self.num_nodes):
                for role in AggregatorRole:
                    pool = self._pool[node_id][role]
                    total_aggregators += len(pool)
                    by_role[role] += len(pool)
                    for instance in pool:
                        if instance.is_idle():
                            idle_count += 1

            return {
                "total_aggregators": total_aggregators,
                "by_role": {role.value: count for role, count in by_role.items()},
                "idle_aggregators": idle_count,
                "creation_count": self._aggregator_creation_count,
                "reuse_count": self._aggregator_reuse_count,
                "termination_count": self._aggregator_termination_count,
                "reuse_ratio": (
                    self._aggregator_reuse_count /
                    (self._aggregator_reuse_count + self._aggregator_creation_count)
                    if (self._aggregator_reuse_count + self._aggregator_creation_count) > 0
                    else 0.0
                ),
                "replan_count": self._replan_count,
            }

    def get_node_aggregators(self, node_id: int) -> Dict[str, Any]:
        """Get detailed info about aggregators on a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary with aggregator details
        """
        with self._lock:
            result = {
                "node_id": node_id,
                "by_role": {},
            }

            for role in AggregatorRole:
                pool = self._pool[node_id][role]
                result["by_role"][role.value] = {
                    "count": len(pool),
                    "idle": sum(1 for inst in pool if inst.is_idle()),
                    "aggregators": [
                        {
                            "id": instance.aggregator.aggregator_id,
                            "age_s": instance.get_age_s(),
                            "idle": instance.is_idle(),
                            "completed_aggregations": instance.completed_aggregations,
                        }
                        for instance in pool
                    ]
                }

            return result
