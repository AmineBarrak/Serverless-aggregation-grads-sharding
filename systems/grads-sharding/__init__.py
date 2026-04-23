"""GradsSharding: Gradient tensor sharding across parallel Lambda aggregators."""

__version__ = "0.1.0"
__author__ = "FL Research Team"

try:
    from .shard_manager import ShardManager, ShardStore
    from .shard_aggregator import ShardAggregator
    from .orchestrator import ShardOrchestrator
    from .server import GradShardingServer
    from .cost_model import CostModel
except ImportError:
    # Fallback for direct execution
    from shard_manager import ShardManager, ShardStore
    from shard_aggregator import ShardAggregator
    from orchestrator import ShardOrchestrator
    from server import GradShardingServer
    from cost_model import CostModel

__all__ = [
    "ShardManager",
    "ShardStore",
    "ShardAggregator",
    "ShardOrchestrator",
    "GradShardingServer",
    "CostModel",
]
