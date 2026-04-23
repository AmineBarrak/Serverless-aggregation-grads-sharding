"""CLI entry point for GradsSharding."""

import argparse
import sys
import os

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import FLConfig

try:
    from .server import GradShardingServer
except ImportError:
    from server import GradShardingServer


def main():
    """Parse arguments and run GradsSharding."""
    parser = argparse.ArgumentParser(
        description="GradsSharding: Gradient tensor sharding across Lambda aggregators"
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=50,
        help="Number of clients",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of training rounds",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=2,
        help="Local training epochs per client",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for local training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b7", "vgg16", "simple_cnn"],
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
        "--num_shards",
        type=int,
        default=4,
        help="Number of gradient shards",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_shards < 1:
        print("Error: num_shards must be >= 1")
        sys.exit(1)

    # Create configuration
    config = FLConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model,
        dataset_name=args.dataset,
        num_shards=args.num_shards,
    )

    # Run server
    server = GradShardingServer(config)
    model, summary = server.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
