"""
Example usage of LIFL components.

This file demonstrates how to use each component of the LIFL system.
Run with: python EXAMPLES.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from shared.config import FLConfig
from shared.models import get_model


def example_1_shared_memory():
    """Example 1: Using SharedMemoryStore for zero-copy communication."""
    print("\n" + "="*60)
    print("Example 1: SharedMemoryStore")
    print("="*60)

    from lifl.shared_memory import SharedMemoryStore

    # Create shared memory store (10GB capacity)
    store = SharedMemoryStore(max_memory_mb=10000)

    # Store some tensors
    weights1 = torch.randn(1000)
    weights2 = torch.randn(1000)

    key1 = store.put("update_1", weights1)
    key2 = store.put("update_2", weights2)

    print(f"Stored tensor 1 with key: {key1}")
    print(f"Stored tensor 2 with key: {key2}")

    # Retrieve tensors (zero-copy reference)
    retrieved = store.get(key1)
    print(f"Retrieved tensor shape: {retrieved.shape}")

    # Check memory usage
    current, peak, percent = store.get_memory_usage()
    print(f"Memory usage: {current:.2f}MB / {peak:.2f}MB peak ({percent:.1f}%)")

    # List all keys
    keys = store.list_keys()
    print(f"Keys in store: {keys}")


def example_2_placement():
    """Example 2: Locality-aware placement of clients."""
    print("\n" + "="*60)
    print("Example 2: LocalityAwarePlacement")
    print("="*60)

    from lifl.placement import LocalityAwarePlacement

    # Create placement engine for 5 nodes, 100 max updates per node
    placement = LocalityAwarePlacement(num_nodes=5, max_service_capacity=100)

    # Place 20 clients using BestFit bin-packing
    client_ids = list(range(20))
    assignment = placement.place_clients(client_ids)

    print("Client Assignment (BestFit):")
    for node_id, clients in assignment.items():
        if clients:
            print(f"  Node {node_id}: {len(clients)} clients - {clients}")

    # Get placement statistics
    stats = placement.get_placement_stats()
    print(f"\nPlacement Statistics:")
    print(f"  Total clients: {stats['total_clients']}")
    print(f"  Load balance score: {stats['load_balance']:.2f}")
    print(f"  Avg load per node: {stats['avg_load_per_node']:.1f}")

    # Plan hierarchy
    hierarchy_plans = placement.plan_hierarchy(
        num_updates_per_node=4,
        max_k_ary=4
    )

    print(f"\nHierarchy Plans:")
    for node_id, plan in hierarchy_plans.items():
        print(f"  Node {node_id}: k={plan.k_ary}, depth={plan.depth}, "
              f"leaves={plan.leaf_aggregators}, middle={plan.middle_aggregators}")

    # EWMA smoothing example
    for node_id in range(5):
        new_load = 50 + node_id * 10  # Varying loads
        smoothed = placement.smooth_load(node_id, new_load)
        print(f"  Node {node_id}: new_load={new_load}, smoothed={smoothed:.1f}")


def example_3_aggregator():
    """Example 3: Using aggregators for model aggregation."""
    print("\n" + "="*60)
    print("Example 3: LIFLAggregator")
    print("="*60)

    from lifl.shared_memory import SharedMemoryStore
    from lifl.aggregator import LIFLAggregator, AggregatorRole

    # Create shared memory and aggregator
    store = SharedMemoryStore(max_memory_mb=1000)
    agg = LIFLAggregator(store)

    # Store some model updates
    update_keys = []
    for i in range(3):
        weights = torch.ones(100) * (i + 1)  # Different weights
        key = store.put(f"update_{i}", weights)
        update_keys.append(key)

    print(f"Created aggregator {agg.aggregator_id}")
    print(f"Input keys to aggregate: {update_keys}")

    # Set role and aggregate
    agg.set_role(AggregatorRole.LEAF)
    result_key = agg.aggregate(update_keys, round_num=1)

    print(f"Aggregated result key: {result_key}")
    print(f"Result should be average of 1, 2, 3 = 2.0")

    result = store.get(result_key)
    print(f"Result value: {result[0].item():.1f}")

    # Get metrics
    metrics = agg.get_metrics()
    print(f"\nAggregation metrics:")
    for metric in metrics:
        print(f"  Time: {metric['total_time_ms']:.2f}ms")
        print(f"  Inputs: {metric['num_inputs']}")
        print(f"  Role: {metric['role']}")


def example_4_autoscaler():
    """Example 4: Aggregator pool management with autoscaling."""
    print("\n" + "="*60)
    print("Example 4: HierarchyAwareAutoscaler")
    print("="*60)

    from lifl.shared_memory import SharedMemoryStore
    from lifl.autoscaler import HierarchyAwareAutoscaler
    from lifl.aggregator import AggregatorRole

    # Create autoscaler
    store = SharedMemoryStore(max_memory_mb=1000)
    autoscaler = HierarchyAwareAutoscaler(store, num_nodes=3, replan_interval_s=10.0)

    print("Creating aggregators...")

    # Get some aggregators
    agg1 = autoscaler.get_aggregator(node_id=0, role=AggregatorRole.LEAF, reuse=False)
    agg2 = autoscaler.get_aggregator(node_id=0, role=AggregatorRole.LEAF, reuse=False)
    agg3 = autoscaler.get_aggregator(node_id=0, role=AggregatorRole.MIDDLE, reuse=False)

    print(f"Created aggregators: {agg1.aggregator_id}, {agg2.aggregator_id}, {agg3.aggregator_id}")

    # Try to reuse
    agg1_reused = autoscaler.get_aggregator(node_id=0, role=AggregatorRole.LEAF, reuse=True)
    print(f"Reused aggregator: {agg1_reused.aggregator_id}")

    # Get pool statistics
    stats = autoscaler.get_pool_statistics()
    print(f"\nAutoscaler Statistics:")
    print(f"  Total aggregators: {stats['total_aggregators']}")
    print(f"  Reuse count: {stats['reuse_count']}")
    print(f"  Creation count: {stats['creation_count']}")
    print(f"  Reuse ratio: {stats['reuse_ratio']:.2%}")


def example_5_gateway():
    """Example 5: Node gateway for update collection."""
    print("\n" + "="*60)
    print("Example 5: NodeGateway")
    print("="*60)

    from lifl.shared_memory import SharedMemoryStore
    from lifl.gateway import NodeGateway

    # Create gateway
    store = SharedMemoryStore(max_memory_mb=1000)
    gateway = NodeGateway(
        node_id=0,
        shared_memory=store,
        num_clients_per_node=4,
        base_cpu_cores=4
    )

    print(f"Created gateway for node 0")

    # Receive some client updates
    for client_id in range(3):
        weights = torch.randn(100)
        success = gateway.receive_update(client_id, round_num=1, weights=weights)
        print(f"  Client {client_id}: {'Success' if success else 'Failed'}")

    # Get pending updates
    pending = gateway.get_pending_updates(round_num=1)
    print(f"Pending updates: {len(pending)} keys")

    # Scale CPU cores based on load
    new_cores = gateway.scale_cpu_cores(load=0.7)
    print(f"Scaled to {new_cores} cores (load=0.7)")

    # Get gateway statistics
    stats = gateway.get_statistics()
    print(f"\nGateway Statistics:")
    print(f"  Updates received: {stats['updates_received']}")
    print(f"  Intra-node transfers: {stats['intra_node_transfers']}")
    print(f"  CPU cores allocated: {stats['cpu_cores_allocated']}")


def example_6_coordinator():
    """Example 6: Coordinator orchestrating FL."""
    print("\n" + "="*60)
    print("Example 6: LIFLCoordinator")
    print("="*60)

    from lifl.coordinator import LIFLCoordinator

    # Create coordinator for 10 clients on 3 nodes
    coordinator = LIFLCoordinator(
        num_clients=10,
        num_nodes=3,
        max_service_capacity=50
    )

    print(f"Coordinator initialized")
    print(f"  Clients: {coordinator.num_clients}")
    print(f"  Nodes: {coordinator.num_nodes}")

    # Check client placement
    print(f"\nClient Placement:")
    for node_id, clients in coordinator.client_placement.items():
        print(f"  Node {node_id}: {len(clients)} clients")

    # Get coordinator statistics
    stats = coordinator.get_coordinator_statistics()
    print(f"\nCoordinator Statistics:")
    print(f"  Current round: {stats['current_round']}")
    print(f"  Placement load balance: {stats['placement_stats']['load_balance']:.2f}")


def example_7_full_fl_loop():
    """Example 7: Full FL training loop."""
    print("\n" + "="*60)
    print("Example 7: Full FL Training Loop (Simplified)")
    print("="*60)

    from lifl.coordinator import LIFLCoordinator
    from shared.models import get_model

    # Create coordinator
    coordinator = LIFLCoordinator(num_clients=4, num_nodes=2, max_service_capacity=20)

    # Get model
    model, param_count = get_model("simple_cnn")
    print(f"Model: simple_cnn ({param_count} parameters)")

    # Simulate 1 FL round
    print(f"\nSimulating FL Round 1:")

    # Step 1: Distribute model
    print("  1. Distributing model...")
    client_models = coordinator.distribute_model(model)
    print(f"     Distributed to {len(client_models)} clients")

    # Step 2: Collect updates (simulated)
    print("  2. Collecting updates...")
    client_updates = {}
    for client_id in range(4):
        # Simulate training by slightly modifying weights
        weights = torch.cat([p.data.flatten() for p in model.parameters()])
        weights = weights + torch.randn_like(weights) * 0.01
        client_updates[client_id] = weights

    update_keys = coordinator.collect_updates(round_num=1, client_updates=client_updates)
    total_updates = sum(len(keys) for keys in update_keys.values())
    print(f"     Collected {total_updates} updates")

    # Step 3: Hierarchical aggregation
    print("  3. Running hierarchical aggregation...")
    aggregated = coordinator.execute_hierarchical_aggregation(
        round_num=1,
        update_keys_by_node=update_keys
    )

    if aggregated is not None:
        print(f"     Aggregation successful: {aggregated.shape}")
    else:
        print("     Aggregation failed")

    # Get statistics
    stats = coordinator.get_coordinator_statistics()
    print(f"\n  Autoscaler stats: {stats['autoscaler_stats']}")


if __name__ == "__main__":
    print("\nLIFL Component Examples")
    print("=" * 60)

    try:
        example_1_shared_memory()
        example_2_placement()
        example_3_aggregator()
        example_4_autoscaler()
        example_5_gateway()
        example_6_coordinator()
        example_7_full_fl_loop()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
