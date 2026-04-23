"""CLI entry point for LIFL - compatible with lambda-fl interface.

Usage:
    python -m lifl.run --num-clients 10 --num-rounds 50 --model simple_cnn --dataset cifar100
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json

# Add shared modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import FLConfig
from shared.metrics import save_metrics
try:
    from .server import LIFLServer
except ImportError:
    from server import LIFLServer

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LIFL: Lightweight, Event-driven Serverless Platform for Federated Learning"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of clients in the federation",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
        help="Number of FL rounds",
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per round",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for SGD",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["resnet18", "efficientnet_b7", "vgg16", "simple_cnn"],
        help="Model architecture",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "femnist", "rvlcdip"],
        help="Dataset name",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of gradient shards (unused in LIFL)",
    )

    parser.add_argument(
        "--memory-limit",
        type=int,
        default=512,
        help="Simulated Lambda memory limit in MB",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for client training in seconds",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save metrics JSON",
    )

    parser.add_argument(
        "--output-stats",
        type=str,
        default=None,
        help="Path to save detailed statistics JSON",
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("LIFL Federated Learning Server")
    logger.info("=" * 60)

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
        memory_limit_mb=args.memory_limit,
        timeout_seconds=args.timeout,
        seed=args.seed,
    )

    logger.info(f"\n{config}")
    logger.info("=" * 60)

    # Create and run server
    try:
        server = LIFLServer(config)
        model, metrics = server.run()

        # Print summary
        logger.info("\n" + metrics.summary())

        # Save metrics if requested
        if args.output_metrics:
            output_path = Path(args.output_metrics)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_metrics(metrics, output_path)
            logger.info(f"Metrics saved to {output_path}")

        # Save detailed statistics if requested
        if args.output_stats:
            output_path = Path(args.output_stats)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stats = server.get_statistics()
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Statistics saved to {output_path}")

        logger.info("\nTraining completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
