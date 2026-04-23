"""CLI entry point for Lambda-FL training."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

from shared.config import FLConfig
try:
    from .server import run_lambda_fl
except ImportError:
    from server import run_lambda_fl


logger = logging.getLogger(__name__)


def main():
    """Parse arguments and run Lambda-FL training."""
    parser = argparse.ArgumentParser(
        description="Lambda-FL: Serverless Aggregation For Federated Learning"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=50,
        help="Total number of clients",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Number of training rounds",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local training epochs per client per round",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for client training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "resnet18", "vgg16", "efficientnet_b7"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "femnist", "rvlcdip"],
        help="Dataset",
    )
    parser.add_argument(
        "--aggregation_goal",
        type=int,
        default=0,
        help="Number of clients to wait for before aggregating (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for metrics",
    )

    args = parser.parse_args()

    # Create config
    config = FLConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model_name=args.model,
        dataset_name=args.dataset,
        aggregation_goal=args.aggregation_goal,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"Lambda-FL: Serverless Aggregation For Federated Learning")
    print(f"{'='*60}")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")

    # Run training
    try:
        model, metrics = run_lambda_fl(config)
        print("\n✓ Training completed successfully")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / f"{job_id}-metrics.json"
    with open(metrics_file, "w") as f:
        # Convert to serializable format
        metrics_serializable = {
            "config": {
                "num_clients": config.num_clients,
                "num_rounds": config.num_rounds,
                "model": config.model_name,
                "dataset": config.dataset_name,
                "k": config.k,
                "learning_rate": config.learning_rate,
            },
            "metrics": metrics,
        }
        json.dump(metrics_serializable, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total training time: {metrics['total_time']:.2f}s")
    print(f"  Num rounds: {len(metrics['rounds'])}")
    if metrics["rounds"]:
        avg_round_time = sum(r["round_time"] for r in metrics["rounds"]) / len(
            metrics["rounds"]
        )
        print(f"  Avg round time: {avg_round_time:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
