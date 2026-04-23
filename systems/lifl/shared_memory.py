"""Shared memory object store for zero-copy intra-node communication.

Simulates a distributed shared memory system where multiple processes can
access model updates without serialization overhead.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class SharedMemoryObject:
    """A shared memory object with metadata."""
    key: str
    tensor: torch.Tensor
    created_at: float
    size_bytes: int
    read_only: bool = True
    access_count: int = 0


class SharedMemoryStore:
    """Thread-safe shared memory store for model updates and aggregation results.

    Features:
    - Immutable objects (read-only after creation) for consistency
    - Memory usage tracking
    - Zero-copy access through reference returns
    - Thread-safe operations
    """

    def __init__(self, max_memory_mb: int = 10000):
        """Initialize the shared memory store.

        Args:
            max_memory_mb: Maximum memory capacity in MB
        """
        self.max_memory_mb = max_memory_mb
        self._store: Dict[str, SharedMemoryObject] = {}
        self._lock = threading.RLock()
        self.total_allocated_mb = 0.0
        self.peak_memory_mb = 0.0
        self._access_history: Dict[str, int] = {}

    def put(self, key: str, tensor: torch.Tensor) -> str:
        """Store a tensor in shared memory.

        Args:
            key: Unique identifier for the object
            tensor: PyTorch tensor to store

        Returns:
            The object key (for reference passing)

        Raises:
            RuntimeError: If memory limit exceeded
        """
        with self._lock:
            if key in self._store:
                self.delete(key)

            # Calculate size in MB
            size_bytes = tensor.element_size() * tensor.nelement()
            size_mb = size_bytes / (1024 * 1024)

            if self.total_allocated_mb + size_mb > self.max_memory_mb:
                raise RuntimeError(
                    f"Memory limit exceeded. Current: {self.total_allocated_mb:.2f}MB, "
                    f"Requested: {size_mb:.2f}MB, Limit: {self.max_memory_mb}MB"
                )

            # Create immutable object
            obj = SharedMemoryObject(
                key=key,
                tensor=tensor.detach().clone(),
                created_at=time.time(),
                size_bytes=size_bytes,
                read_only=True,
                access_count=0
            )

            self._store[key] = obj
            self.total_allocated_mb += size_mb
            self.peak_memory_mb = max(self.peak_memory_mb, self.total_allocated_mb)
            self._access_history[key] = 0

            logger.debug(f"Stored {key} ({size_mb:.2f}MB). Total: {self.total_allocated_mb:.2f}MB")
            return key

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a tensor from shared memory (zero-copy reference).

        Args:
            key: Object identifier

        Returns:
            The tensor or None if not found
        """
        with self._lock:
            if key not in self._store:
                return None

            obj = self._store[key]
            obj.access_count += 1
            self._access_history[key] = self._access_history.get(key, 0) + 1

            # In real zero-copy, this would be a reference.
            # For simulation, we return the tensor directly (still avoids network transfer)
            return obj.tensor

    def delete(self, key: str) -> bool:
        """Remove an object from shared memory.

        Args:
            key: Object identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key not in self._store:
                return False

            obj = self._store.pop(key)
            size_mb = obj.size_bytes / (1024 * 1024)
            self.total_allocated_mb -= size_mb

            logger.debug(f"Deleted {key} ({size_mb:.2f}MB). Total: {self.total_allocated_mb:.2f}MB")
            return True

    def list_keys(self) -> List[str]:
        """Get all keys currently in the store.

        Returns:
            List of object keys
        """
        with self._lock:
            return list(self._store.keys())

    def get_memory_usage(self) -> Tuple[float, float, float]:
        """Get current memory usage metrics.

        Returns:
            Tuple of (current_mb, peak_mb, percent_of_limit)
        """
        with self._lock:
            percent = (self.total_allocated_mb / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0
            return self.total_allocated_mb, self.peak_memory_mb, percent

    def get_object_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a stored object.

        Args:
            key: Object identifier

        Returns:
            Dictionary with object info or None if not found
        """
        with self._lock:
            if key not in self._store:
                return None

            obj = self._store[key]
            return {
                "key": obj.key,
                "shape": tuple(obj.tensor.shape),
                "dtype": str(obj.dtype),
                "size_mb": obj.size_bytes / (1024 * 1024),
                "created_at": obj.created_at,
                "access_count": obj.access_count,
                "read_only": obj.read_only,
            }

    def clear(self) -> None:
        """Clear all objects from the store."""
        with self._lock:
            self._store.clear()
            self.total_allocated_mb = 0.0
            self._access_history.clear()
            logger.info("Shared memory store cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive store statistics.

        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            total_accesses = sum(self._access_history.values())
            return {
                "total_objects": len(self._store),
                "current_memory_mb": self.total_allocated_mb,
                "peak_memory_mb": self.peak_memory_mb,
                "memory_limit_mb": self.max_memory_mb,
                "total_accesses": total_accesses,
                "objects": {
                    key: {
                        "size_mb": obj.size_bytes / (1024 * 1024),
                        "accesses": self._access_history.get(key, 0),
                    }
                    for key, obj in self._store.items()
                }
            }
