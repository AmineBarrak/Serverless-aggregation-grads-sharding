"""Simple example demonstrating Lambda-FL components.

This example shows how to use the core Lambda-FL components
without a full training loop.
"""

import sys
import os

# Setup path - add parent directory so we can import lambda_fl
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from message_queue import SimulatedKafka

try:
    from aggregator import LeafAggregator, IntermediateAggregator
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    LeafAggregator = None
    IntermediateAggregator = None
    print("Note: PyTorch not available. Showing message queue example only.\n")


def example_message_queue():
    """Example: Using the message queue for communication."""
    print("=" * 60)
    print("Example 1: Message Queue (SimulatedKafka)")
    print("=" * 60)

    queue = SimulatedKafka()

    # Publishers (clients) publish gradients
    for client_id in range(3):
        message = {"client_id": client_id, "gradient": f"grad_{client_id}"}
        queue.publish("job-0-parties", message)
        print(f"  Client {client_id} published gradient")

    # Check queue size
    size = queue.get_topic_size("job-0-parties")
    print(f"\nQueue 'job-0-parties' has {size} messages")

    # Consumer (aggregator) consumes messages
    messages = queue.consume("job-0-parties", count=3, timeout=1.0)
    print(f"\nAggregator consumed {len(messages)} messages:")
    for msg in messages:
        print(f"  {msg}")

    print()


def example_aggregators():
    """Example: Leaf and Intermediate Aggregators."""
    if not TORCH_AVAILABLE:
        print("Skipping aggregator example (PyTorch required)")
        return

    print("=" * 60)
    print("Example 2: Leaf and Intermediate Aggregators")
    print("=" * 60)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model = SimpleModel()

    # Simulate k=5 clients sending gradients
    print("\nSimulating 5 clients computing gradients...")
    client_updates = []
    for client_id in range(5):
        # Each client has a gradient dictionary
        grads = {
            name: torch.randn_like(param) * 0.01
            for name, param in model.named_parameters()
        }
        client_updates.append(grads)
        print(f"  Client {client_id}: {len(grads)} parameters")

    # Leaf Aggregator: aggregate 5 client updates
    print("\nLeaf Aggregator: Processing 5 client updates...")
    leaf_agg = LeafAggregator(agg_id=0, k=5)
    sum_grads, count, metadata = leaf_agg.aggregate(client_updates)

    print(f"  Aggregated {count} updates")
    print(f"  Result: {len(sum_grads)} parameter sums")
    print(f"  Latency: {metadata['latency']*1000:.2f}ms")

    # Intermediate Aggregator: aggregate leaf results
    print("\nIntermediate Aggregator: Processing leaf results...")
    ia_agg = IntermediateAggregator(agg_id=0, num_leaves=1)
    final_grads, ia_metadata = ia_agg.aggregate([(sum_grads, count)])

    print(f"  Final averaged {len(final_grads)} parameters")
    print(f"  Total clients: {ia_metadata['total_clients']}")
    print(f"  Latency: {ia_metadata['latency']*1000:.2f}ms")

    print()


def example_coordinator():
    """Example: Coordinator topology."""
    if not TORCH_AVAILABLE:
        print("Skipping coordinator example (PyTorch required)")
        return

    from coordinator import LambdaFLCoordinator

    print("=" * 60)
    print("Example 3: Coordinator Tree Topology")
    print("=" * 60)

    # Create coordinator for 50 clients with k=5
    coordinator = LambdaFLCoordinator(
        num_parties=50,
        k=5,
        job_id="example-job",
    )

    tree_info = coordinator.get_tree_info()
    print(f"\nConfiguration:")
    print(f"  Total clients: {tree_info['num_parties']}")
    print(f"  Clients per leaf: {tree_info['k']}")
    print(f"  Leaf aggregators: {tree_info['num_leaf_aggregators']}")
    print(f"  Intermediate aggregators: {tree_info['num_intermediate_aggregators']}")

    coordinator.shutdown()
    print()


if __name__ == "__main__":
    print("\n")
    print("Lambda-FL Component Examples")
    print("-" * 60)
    print()

    example_message_queue()
    example_aggregators()
    example_coordinator()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print()
