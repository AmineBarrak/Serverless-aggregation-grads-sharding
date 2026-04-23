"""Gradient shard management: splitting, merging, and storage operations."""

import time
from typing import List, Tuple, Dict, Any
import torch
import numpy as np


class ShardManager:
    """Manages gradient sharding: splitting and merging operations."""

    def __init__(self):
        """Initialize the shard manager."""
        self.shard_sizes = {}
        self.split_times = []
        self.merge_times = []

    def get_shard_assignments(
        self, model_size: int, num_shards: int
    ) -> List[Tuple[int, int]]:
        """
        Get shard assignments as (start, end) index pairs.

        Args:
            model_size: Total number of parameters in the model
            num_shards: Number of shards to split into

        Returns:
            List of (start, end) tuples for each shard
        """
        shard_size = model_size // num_shards
        remainder = model_size % num_shards

        assignments = []
        current_idx = 0

        for shard_id in range(num_shards):
            # Distribute remainder across first shards
            size = shard_size + (1 if shard_id < remainder else 0)
            end_idx = current_idx + size
            assignments.append((current_idx, end_idx))
            current_idx = end_idx

        return assignments

    def split_gradient(self, gradient_tensor: torch.Tensor, num_shards: int) -> List[torch.Tensor]:
        """
        Split gradient tensor into shards.

        Args:
            gradient_tensor: Full gradient tensor (flattened)
            num_shards: Number of shards to create

        Returns:
            List of shard tensors
        """
        start_time = time.time()

        model_size = gradient_tensor.numel()
        assignments = self.get_shard_assignments(model_size, num_shards)

        shards = []
        for shard_id, (start, end) in enumerate(assignments):
            shard = gradient_tensor[start:end].clone()
            shards.append(shard)
            self.shard_sizes[shard_id] = shard.numel()

        elapsed = time.time() - start_time
        self.split_times.append(elapsed)

        return shards

    def merge_shards(self, shards: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge shards back into full gradient tensor.

        Args:
            shards: List of shard tensors

        Returns:
            Full gradient tensor
        """
        start_time = time.time()

        merged = torch.cat(shards, dim=0)

        elapsed = time.time() - start_time
        self.merge_times.append(elapsed)

        return merged

    def get_shard_memory_mb(self, shard_id: int) -> float:
        """Get memory usage for a shard in MB."""
        num_params = self.shard_sizes.get(shard_id, 0)
        # Assume float32 (4 bytes per parameter)
        bytes_per_param = 4
        total_bytes = num_params * bytes_per_param
        return total_bytes / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about shard operations."""
        return {
            "avg_split_time_s": np.mean(self.split_times) if self.split_times else 0,
            "avg_merge_time_s": np.mean(self.merge_times) if self.merge_times else 0,
            "total_split_time_s": sum(self.split_times),
            "total_merge_time_s": sum(self.merge_times),
            "shard_sizes": self.shard_sizes,
        }


class ShardStore:
    """Simulates S3 storage for gradient shards."""

    def __init__(self):
        """Initialize the shard store."""
        self.store: Dict[str, torch.Tensor] = {}
        self.upload_times = []
        self.download_times = []
        self.upload_sizes = []  # bytes
        self.download_sizes = []  # bytes

    def _make_key(self, client_id: int, shard_id: int, round_num: int) -> str:
        """Create a storage key for a shard."""
        return f"client_{client_id}/round_{round_num}/shard_{shard_id}"

    def upload_shard(
        self, client_id: int, shard_id: int, round_num: int, tensor: torch.Tensor
    ) -> float:
        """
        Upload a shard to storage.

        Args:
            client_id: Client ID
            shard_id: Shard ID
            round_num: Round number
            tensor: Tensor to upload

        Returns:
            Upload time in seconds
        """
        start_time = time.time()

        key = self._make_key(client_id, shard_id, round_num)
        self.store[key] = tensor.clone()

        elapsed = time.time() - start_time
        self.upload_times.append(elapsed)
        self.upload_sizes.append(tensor.numel() * 4)

        return elapsed

    def download_shards(
        self, shard_id: int, round_num: int, client_ids: List[int]
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Download all client shards for a given shard_id.

        Args:
            shard_id: Shard ID to retrieve
            round_num: Round number
            client_ids: List of client IDs

        Returns:
            Tuple of (list of tensors, total download time)
        """
        start_time = time.time()

        shards = []
        total_size = 0

        for client_id in client_ids:
            key = self._make_key(client_id, shard_id, round_num)
            if key in self.store:
                tensor = self.store[key].clone()
                shards.append(tensor)
                total_size += tensor.numel() * 4
            else:
                raise KeyError(f"Shard not found: {key}")

        elapsed = time.time() - start_time
        self.download_times.append(elapsed)
        self.download_sizes.append(total_size)

        return shards, elapsed

    def clear_round(self, round_num: int) -> None:
        """Clear all shards for a given round."""
        keys_to_delete = [k for k in self.store.keys() if f"round_{round_num}" in k]
        for key in keys_to_delete:
            del self.store[key]

    def get_storage_size_mb(self) -> float:
        """Get total storage size in MB."""
        total_bytes = sum(t.numel() * 4 for t in self.store.values())
        return total_bytes / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about S3 operations."""
        return {
            "total_uploads": len(self.upload_times),
            "total_downloads": len(self.download_times),
            "avg_upload_time_s": np.mean(self.upload_times) if self.upload_times else 0,
            "avg_download_time_s": np.mean(self.download_times) if self.download_times else 0,
            "total_upload_time_s": sum(self.upload_times),
            "total_download_time_s": sum(self.download_times),
            "total_upload_size_mb": sum(self.upload_sizes) / (1024 * 1024),
            "total_download_size_mb": sum(self.download_sizes) / (1024 * 1024),
            "current_storage_mb": self.get_storage_size_mb(),
        }
