"""LIFL: A Lightweight, Event-driven Serverless Platform for Federated Learning.

MLSys 2024 paper implementation featuring:
- Hierarchical aggregation with shared memory for intra-node zero-copy communication
- Gateway-based model update collection with protocol processing
- Stateless, reusable aggregators with eager aggregation
- eBPF-based lightweight sidecar simulation
- Locality-aware placement using BestFit bin-packing
- Dynamic hierarchy planning with EWMA smoothing
- Aggregator reuse and promotion for warm starts
"""

__version__ = "1.0.0"
__author__ = "MLSys 2024"

from .shared_memory import SharedMemoryStore
from .gateway import NodeGateway
from .aggregator import LIFLAggregator
from .placement import LocalityAwarePlacement
from .autoscaler import HierarchyAwareAutoscaler
from .coordinator import LIFLCoordinator
from .server import LIFLServer

__all__ = [
    "SharedMemoryStore",
    "NodeGateway",
    "LIFLAggregator",
    "LocalityAwarePlacement",
    "HierarchyAwareAutoscaler",
    "LIFLCoordinator",
    "LIFLServer",
]
