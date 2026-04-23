"""
Unified experiment runner for Research Questions 1-3.

Usage:
    python -m experiments.run_experiment --rq 1              # Run RQ1 only
    python -m experiments.run_experiment --rq all             # Run all RQs
    python -m experiments.run_experiment --rq 1 --quick       # Run RQ1 in quick mode
    python -m experiments.run_experiment --rq 2 --part a      # Run RQ2 Part A (HPC) only
    python -m experiments.run_experiment --rq 2 --part b      # Run RQ2 Part B (analytical) only

Supports:
    --rq <n>          : Research question number(s): 1, 2, or "all"
    --quick           : Quick mode (fewer clients, rounds, use SimpleCNN)
    --part <a|b|both> : For RQ2: which part to run (default: both)
    --skip-plotting   : Skip plotting step
    --no-report       : Don't generate summary report

All results saved to: experiments/results/{rq_name}/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import rq1_shard_ablation, rq2_cross_architecture, rq3_headtohead


def main():
    """Parse arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description="Run FL system experiments for Research Questions 1-2"
    )

    parser.add_argument(
        "--rq",
        type=str,
        default="all",
        help="RQ to run: 1, 2, or 'all' (comma-separated also supported)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer rounds and clients for testing",
    )

    parser.add_argument(
        "--part",
        type=str,
        default="both",
        choices=["a", "b", "both"],
        help="For RQ2: 'a' for HPC measured, 'b' for analytical, 'both' for all",
    )

    parser.add_argument(
        "--skip-plotting",
        action="store_true",
        help="Skip plotting step",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't generate summary report",
    )

    args = parser.parse_args()

    # Parse RQ numbers
    if args.rq.lower() == "all":
        rq_numbers = [1, 2, 3]
    else:
        try:
            rq_numbers = [int(x.strip()) for x in args.rq.split(",")]
            for rq in rq_numbers:
                if rq not in [1, 2, 3]:
                    print(f"Error: Invalid RQ number {rq}. Must be 1, 2, or 3.")
                    sys.exit(1)
        except ValueError:
            print(f"Error: Invalid RQ specification '{args.rq}'")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SYSTEMS EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Running RQs: {rq_numbers}")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)

    all_results = {}

    if 1 in rq_numbers:
        print("\n[1/3] Running RQ1 - Shard Ablation Study...")
        try:
            results = rq1_shard_ablation.run_rq1(quick=args.quick)
            all_results["rq1"] = results
            print("✓ RQ1 completed")
        except Exception as e:
            print(f"✗ RQ1 failed: {e}")
            import traceback
            traceback.print_exc()

    if 2 in rq_numbers:
        print("\n[2/3] Running RQ2 - Cross-Architecture Comparison...")
        try:
            results = rq2_cross_architecture.run_rq2(
                quick=args.quick, part=args.part
            )
            all_results["rq2"] = results
            print("✓ RQ2 completed")
        except Exception as e:
            print(f"✗ RQ2 failed: {e}")
            import traceback
            traceback.print_exc()

    if 3 in rq_numbers:
        print("\n[3/3] Running RQ3 - Head-to-Head Comparison...")
        try:
            results = rq3_headtohead.run_rq3(quick=args.quick)
            all_results["rq3"] = results
            print("✓ RQ3 completed")
        except Exception as e:
            print(f"✗ RQ3 failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {PROJECT_ROOT}/experiments/results/")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
