"""AWS Lambda cost estimation for GradsSharding.

Uses Lambda-compute-only pricing for fair comparison with Lambda-FL and LIFL.
All three systems are charged identically: memory_GB * seconds * $0.0000166667.
"""

from typing import Dict, Any


class CostModel:
    """Estimates AWS Lambda compute cost for GradsSharding execution."""

    # AWS Lambda Pricing (as of 2024)
    LAMBDA_COST_PER_GB_SECOND = 0.0000166667  # $0.0000166667 per GB-second

    def __init__(self):
        """Initialize the cost model."""
        self.round_costs = []

    @staticmethod
    def estimate_lambda_cost(
        memory_mb: float,
        execution_time_s: float,
    ) -> float:
        """
        Estimate Lambda execution cost.

        Args:
            memory_mb: Memory allocation in MB
            execution_time_s: Execution time in seconds

        Returns:
            Cost in USD
        """
        memory_gb = memory_mb / 1024
        gb_seconds = memory_gb * execution_time_s
        return gb_seconds * CostModel.LAMBDA_COST_PER_GB_SECOND

    @staticmethod
    def estimate_round_cost(
        num_shards: int,
        aggregation_time_s: float,
        memory_mb: float,
    ) -> Dict[str, float]:
        """
        Estimate total Lambda compute cost for one aggregation round.

        Args:
            num_shards: Number of parallel shard aggregators
            aggregation_time_s: Max aggregation time per shard in seconds
            memory_mb: Peak memory allocated per Lambda in MB

        Returns:
            Dictionary with cost breakdown
        """
        # Lambda costs: num_shards parallel invocations
        lambda_cost = CostModel.estimate_lambda_cost(memory_mb, aggregation_time_s) * num_shards

        return {
            "lambda_cost_usd": lambda_cost,
            "total_cost_usd": lambda_cost,
        }

    def add_round_cost(self, cost_breakdown: Dict[str, float]) -> None:
        """Add a round cost to history."""
        self.round_costs.append(cost_breakdown)

    def get_total_cost(self) -> Dict[str, float]:
        """Get aggregated costs across all rounds."""
        if not self.round_costs:
            return {
                "lambda_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            }

        return {
            "lambda_cost_usd": sum(c["lambda_cost_usd"] for c in self.round_costs),
            "total_cost_usd": sum(c["total_cost_usd"] for c in self.round_costs),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics."""
        total = self.get_total_cost()
        return {
            "num_rounds": len(self.round_costs),
            "total_costs": total,
            "avg_cost_per_round_usd": total["total_cost_usd"] / len(self.round_costs)
            if self.round_costs
            else 0.0,
        }
