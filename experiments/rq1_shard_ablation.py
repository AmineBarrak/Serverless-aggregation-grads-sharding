"""
RQ1 - Shard Ablation Study: Optimal K Across Model Scales.

Goal: Determine how the number of shards K affects aggregation latency,
      memory, and cost for GradsSharding across different model sizes.

Tests:
  - Shard counts: M in {1, 2, 4, 8, 16}
  - Models: ResNet-18 (45MB), VGG-16 (528MB)
  - Fixed: N=20 clients, 10 rounds
  - Metrics: per-aggregator memory, aggregation latency, total cost

The key insight: for small models (ResNet-18), varying K barely matters.
For large models (VGG-16), K controls whether aggregation fits in Lambda's
10GB limit and determines the speed/cost tradeoff.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runner import run_system


def run_rq1(quick: bool = False) -> Dict[str, Any]:
    """Run RQ1 - Shard ablation study across model sizes."""
    print("\n" + "=" * 70)
    print("RQ1: Shard Ablation Study — Optimal K Across Model Scales")
    print("=" * 70)

    # Experiment parameters
    if quick:
        shard_counts = [1, 2, 4]
        model_configs = [
            {"model": "simple_cnn", "dataset": "cifar100", "batch_size": 32},
        ]
        num_clients = 4
        num_rounds = 2
    else:
        shard_counts = [1, 2, 4, 8, 16]
        model_configs = [
            {"model": "resnet18", "dataset": "cifar100", "batch_size": 32},
            {"model": "vgg16", "dataset": "rvlcdip", "batch_size": 8},
        ]
        num_clients = 20
        num_rounds = 10

    output_dir = PROJECT_ROOT / "experiments" / "results" / "rq1_shard_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "quick_mode": quick,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "shard_counts": shard_counts,
            "models": [m["model"] for m in model_configs],
        }
    }

    for mc in model_configs:
        model_name = mc["model"]
        dataset = mc["dataset"]
        batch_size = mc["batch_size"]
        model_results = {}

        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name} | Dataset: {dataset} | BS: {batch_size}")
        print(f"{'─' * 60}")

        for M in shard_counts:
            key = f"{model_name}_M{M}"
            print(f"\n--- GradsSharding: M={M} shards, model={model_name} ---")
            model_results[f"shards_{M}"] = run_system(
                system="grads-sharding",
                num_clients=num_clients,
                num_rounds=num_rounds,
                model_name=model_name,
                dataset_name=dataset,
                num_shards=M,
                batch_size=batch_size,
                output_file=str(output_dir / f"{model_name}_M{M}.json"),
            )

        results[model_name] = model_results

    # Save aggregated results
    agg_file = output_dir / "rq1_results.json"
    with open(agg_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRQ1 results saved to {agg_file}")
    return results


if __name__ == "__main__":
    run_rq1(quick="--quick" in sys.argv)
