"""Example usage of GradsSharding components."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from shard_manager import ShardManager, ShardStore
from shard_aggregator import ShardAggregator
from orchestrator import ShardOrchestrator
from cost_model import CostModel


def example_shard_manager():
    """Demonstrate ShardManager functionality."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Shard Manager")
    print("="*80)

    manager = ShardManager()

    # Example gradient tensor (1B parameters)
    model_size = 1_000_000_000
    gradient = torch.randn(model_size)

    # Split into shards
    num_shards = 4
    shards = manager.split_gradient(gradient, num_shards)

    print(f"Original gradient size: {gradient.numel():,} parameters")
    print(f"Number of shards: {num_shards}")
    print(f"Shard sizes: {[s.numel() for s in shards]}")

    # Merge shards
    merged = manager.merge_shards(shards)
    print(f"Merged gradient size: {merged.numel():,} parameters")
    print(f"Merge successful: {torch.allclose(gradient, merged)}")

    # Shard assignments
    assignments = manager.get_shard_assignments(model_size, num_shards)
    print(f"\nShard assignments (start, end):")
    for i, (start, end) in enumerate(assignments):
        print(f"  Shard {i}: [{start:,}, {end:,}) - {end-start:,} params")

    return manager


def example_shard_store():
    """Demonstrate ShardStore (S3 simulation)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Shard Store (S3 Simulation)")
    print("="*80)

    store = ShardStore()

    # Simulate 3 clients uploading shards
    num_clients = 3
    num_shards = 4
    shard_size = 250_000

    print(f"Uploading {num_clients} clients x {num_shards} shards...")
    for client_id in range(num_clients):
        for shard_id in range(num_shards):
            shard = torch.randn(shard_size)
            store.upload_shard(client_id, shard_id, round_num=0, tensor=shard)

    print(f"Total storage: {store.get_storage_size_mb():.2f} MB")

    # Simulate download
    print(f"\nDownloading shard 0 from all clients...")
    shards, download_time = store.download_shards(
        shard_id=0,
        round_num=0,
        client_ids=list(range(num_clients)),
    )

    print(f"Downloaded {len(shards)} shards")
    print(f"Download time: {download_time:.3f}s")

    # Cleanup
    store.clear_round(round_num=0)
    print(f"After cleanup, storage: {store.get_storage_size_mb():.2f} MB")

    # Stats
    stats = store.get_stats()
    print(f"\nStore Statistics:")
    print(f"  Total uploads: {stats['total_uploads']}")
    print(f"  Total downloads: {stats['total_downloads']}")
    print(f"  Avg upload time: {stats['avg_upload_time_s']:.3f}s")
    print(f"  Avg download time: {stats['avg_download_time_s']:.3f}s")

    return store


def example_shard_aggregator():
    """Demonstrate ShardAggregator."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Shard Aggregator")
    print("="*80)

    # Simulate 5 clients' shard
    num_clients = 5
    shard_size = 250_000

    client_shards = [torch.randn(shard_size) for _ in range(num_clients)]

    aggregator = ShardAggregator(shard_id=0, num_clients=num_clients)

    print(f"Aggregating {num_clients} shards (size: {shard_size:,} each)")

    aggregated, metrics = aggregator.aggregate(client_shards)

    print(f"Aggregated shard size: {aggregated.numel():,} parameters")
    print(f"Total execution time: {metrics['total_execution_time_s']:.3f}s")
    print(f"Cold start latency: {metrics['cold_start_latency_s']:.3f}s")
    print(f"Aggregation compute time: {metrics['aggregation_compute_time_s']:.3f}s")
    print(f"Peak memory: {metrics['peak_memory_mb']:.1f} MB")

    return aggregator


def example_orchestrator():
    """Demonstrate ShardOrchestrator."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Shard Orchestrator")
    print("="*80)

    orchestrator = ShardOrchestrator(num_shards=4)

    # Simulate 10 clients
    num_clients = 10
    model_size = 1_000_000  # 1M parameters

    client_gradients = [torch.randn(model_size) for _ in range(num_clients)]

    print(f"Orchestrating round with {num_clients} clients, {4} shards...")
    print(f"Total gradient size: {model_size:,} parameters per client")

    aggregated, round_metrics = orchestrator.orchestrate_round(
        client_gradients,
        num_shards=4,
        max_workers=4,
    )

    print(f"\nRound Results:")
    print(f"  Total round time: {round_metrics['total_round_time_s']:.3f}s")
    print(f"  Client shard upload time: {round_metrics['client_shard_upload_time_s']:.3f}s")
    print(f"  Parallel aggregation time: {round_metrics['parallel_aggregation_time_s']:.3f}s")
    print(f"  Max aggregator exec time: {round_metrics['max_aggregator_execution_time_s']:.3f}s")
    print(f"  Merge time: {round_metrics['merge_time_s']:.3f}s")
    print(f"  Max peak memory per function: {round_metrics['max_peak_memory_mb']:.1f} MB")
    print(f"  Step function transitions: {round_metrics['step_function_transitions']}")

    return orchestrator


def example_cost_model():
    """Demonstrate CostModel."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Cost Model")
    print("="*80)

    cost_model = CostModel()

    # Estimate costs for a round
    num_clients = 50
    num_shards = 4
    shard_size_bytes = 250_000_000  # 250 MB per shard
    aggregation_time_s = 5.0  # 5 seconds per aggregator
    memory_mb = 512
    num_transitions = 10  # Step function transitions

    cost = CostModel.estimate_round_cost(
        num_clients=num_clients,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        aggregation_time_s=aggregation_time_s,
        memory_mb=memory_mb,
        num_transitions=num_transitions,
    )

    cost_model.add_round_cost(cost)

    print(f"Cost Estimation for 1 Round:")
    print(f"  Num clients: {num_clients}")
    print(f"  Num shards: {num_shards}")
    print(f"  Shard size: {shard_size_bytes / (1024**2):.1f} MB")
    print(f"  Aggregation time: {aggregation_time_s:.2f}s")
    print(f"  Memory: {memory_mb} MB")
    print(f"\nCost Breakdown:")
    print(f"  Lambda cost: ${cost['lambda_cost_usd']:.6f}")
    print(f"  S3 PUT cost: ${cost['s3_put_cost_usd']:.6f}")
    print(f"  S3 GET cost: ${cost['s3_get_cost_usd']:.6f}")
    print(f"  S3 Storage cost: ${cost['s3_storage_cost_usd']:.6f}")
    print(f"  Step Functions cost: ${cost['step_function_cost_usd']:.6f}")
    print(f"  Total cost: ${cost['total_cost_usd']:.6f}")

    return cost_model


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GradsSharding Implementation Examples")
    print("="*80)

    example_shard_manager()
    example_shard_store()
    example_shard_aggregator()
    example_orchestrator()
    example_cost_model()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
