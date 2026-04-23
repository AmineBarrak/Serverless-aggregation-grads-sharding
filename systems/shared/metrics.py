"""Metrics collection and reporting for federated learning experiments.

Tracks latency, memory usage, compute costs, and other performance metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class MetricsCollector:
    """Collects performance metrics for FL experiments.

    Attributes:
        round_latency: Time to complete a full FL round (seconds)
        aggregation_latency: Time to aggregate gradients (seconds)
        peak_memory_mb: Peak memory usage per function (Dict[str, float])
        cold_start_count: Number of Lambda cold starts encountered
        total_lambda_seconds: Simulated total Lambda compute-seconds
        cost_usd: Estimated cost in USD
    """

    round_latency: float = 0.0
    aggregation_latency: float = 0.0
    peak_memory_mb: Dict[str, float] = field(default_factory=dict)
    cold_start_count: int = 0
    total_lambda_seconds: float = 0.0
    cost_usd: float = 0.0

    # Additional tracking
    per_round_metrics: list[Dict[str, Any]] = field(default_factory=list)

    def add_round_metric(
        self,
        round_num: int,
        latency: float,
        aggregation_latency: float,
        memory_mb: float,
        cold_start: bool = False,
    ) -> None:
        """Record metrics for a single FL round.

        Args:
            round_num: Round number
            latency: Round latency in seconds
            aggregation_latency: Aggregation latency in seconds
            memory_mb: Peak memory usage in MB
            cold_start: Whether a cold start occurred
        """
        self.per_round_metrics.append(
            {
                "round": round_num,
                "latency_s": latency,
                "aggregation_latency_s": aggregation_latency,
                "peak_memory_mb": memory_mb,
                "cold_start": cold_start,
            }
        )

    def add_function_memory(self, func_name: str, memory_mb: float) -> None:
        """Record peak memory for a specific function.

        Args:
            func_name: Name of the function
            memory_mb: Peak memory usage in MB
        """
        if func_name not in self.peak_memory_mb:
            self.peak_memory_mb[func_name] = memory_mb
        else:
            self.peak_memory_mb[func_name] = max(
                self.peak_memory_mb[func_name], memory_mb
            )

    def compute_cost(self, gb_memory: int = 512) -> float:
        """Compute estimated AWS Lambda cost.

        Cost = (GB * seconds * $0.0000166667 per GB-second)
        Assumes 512 MB by default.

        Args:
            gb_memory: Memory allocation in MB (default 512)

        Returns:
            Estimated cost in USD
        """
        gb = gb_memory / 1024
        cost_per_gb_second = 0.0000166667
        self.cost_usd = self.total_lambda_seconds * gb * cost_per_gb_second
        return self.cost_usd

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return asdict(self)

    def summary(self) -> str:
        """Return a human-readable summary of metrics."""
        lines = [
            "=== Metrics Summary ===",
            f"Round Latency: {self.round_latency:.2f}s",
            f"Aggregation Latency: {self.aggregation_latency:.2f}s",
            f"Cold Starts: {self.cold_start_count}",
            f"Total Lambda-seconds: {self.total_lambda_seconds:.2f}",
            f"Estimated Cost: ${self.cost_usd:.6f}",
            f"Peak Memory (MB): {self.peak_memory_mb}",
        ]
        return "\n".join(lines)


def save_metrics(collector: MetricsCollector, path: str | Path) -> None:
    """Save metrics to a JSON file.

    Args:
        collector: MetricsCollector instance
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(collector.to_dict(), f, indent=2)

    print(f"Metrics saved to {path}")


def load_metrics(path: str | Path) -> MetricsCollector:
    """Load metrics from a JSON file.

    Args:
        path: Input file path

    Returns:
        MetricsCollector instance
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    return MetricsCollector(
        round_latency=data.get("round_latency", 0.0),
        aggregation_latency=data.get("aggregation_latency", 0.0),
        peak_memory_mb=data.get("peak_memory_mb", {}),
        cold_start_count=data.get("cold_start_count", 0),
        total_lambda_seconds=data.get("total_lambda_seconds", 0.0),
        cost_usd=data.get("cost_usd", 0.0),
        per_round_metrics=data.get("per_round_metrics", []),
    )
