"""Utility functions for simulating Lambda execution and tensor operations.

Includes Lambda simulation (cold starts, network transfer), tensor operations,
and gradient sharding utilities.
"""

from typing import Callable, Any, TypeVar, List
import time
import random
import psutil
import torch
import torch.nn as nn

T = TypeVar("T")

# Track function invocations for cold start simulation
_function_invocation_count: dict[str, int] = {}
_random_seed = 42


def _get_cold_start_delay() -> float:
    """Simulate a Lambda cold start delay (0.5-2.0 seconds)."""
    return random.uniform(0.5, 2.0)


def simulate_lambda_invoke(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: dict = None,
    func_name: str = "",
    memory_mb: int = 512,
) -> tuple[T, dict[str, Any]]:
    """Simulate an AWS Lambda function invocation with cold start tracking.

    Tracks execution time, peak memory usage, and cold start delay for first invocation.

    Args:
        func: Function to invoke
        args: Positional arguments for func
        kwargs: Keyword arguments for func (default empty dict)
        func_name: Identifier for cold start tracking (use func.__name__ if not provided)
        memory_mb: Memory allocation (affects cost calculation)

    Returns:
        Tuple of (result, metadata_dict) where metadata contains:
        - "execution_time_s": Function execution time
        - "peak_memory_mb": Peak memory during execution
        - "cold_start": Whether a cold start occurred
        - "cold_start_delay_s": Cold start delay if applicable
    """
    if kwargs is None:
        kwargs = {}

    if not func_name:
        func_name = getattr(func, "__name__", "unknown_function")

    # Check if this is first invocation (cold start)
    is_cold_start = func_name not in _function_invocation_count
    if is_cold_start:
        _function_invocation_count[func_name] = 0
        cold_start_delay = _get_cold_start_delay()
    else:
        cold_start_delay = 0.0

    _function_invocation_count[func_name] += 1

    # Measure execution time and peak memory
    process = psutil.Process()
    start_time = time.time()
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    try:
        result = func(*args, **kwargs)
    finally:
        mem_after = process.memory_info().rss / (1024 * 1024)
        end_time = time.time()

    execution_time = end_time - start_time
    peak_memory = max(mem_before, mem_after)

    metadata = {
        "execution_time_s": execution_time,
        "peak_memory_mb": peak_memory,
        "cold_start": is_cold_start,
        "cold_start_delay_s": cold_start_delay if is_cold_start else 0.0,
        "total_time_s": execution_time + cold_start_delay,
        "memory_mb": memory_mb,
        "invocation_count": _function_invocation_count[func_name],
    }

    return result, metadata


def simulate_network_transfer(
    tensor_size_bytes: int, bandwidth_mbps: float = 1000.0
) -> float:
    """Simulate network transfer time for a tensor.

    Args:
        tensor_size_bytes: Size of data in bytes
        bandwidth_mbps: Network bandwidth in Mbps (default 1000 = 1 Gbps)

    Returns:
        Transfer time in seconds
    """
    bandwidth_bytes_per_sec = (bandwidth_mbps / 8) * (1024 * 1024)
    transfer_time = tensor_size_bytes / bandwidth_bytes_per_sec
    return transfer_time


def flatten_model_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor.

    Args:
        model: PyTorch model

    Returns:
        Flattened parameter tensor
    """
    return torch.cat([p.data.flatten() for p in model.parameters()])


def unflatten_model_params(flat_tensor: torch.Tensor, model: nn.Module) -> None:
    """Unflatten a 1D tensor and set model parameters.

    Args:
        flat_tensor: Flattened parameter tensor
        model: PyTorch model to update (in-place)
    """
    offset = 0
    for param in model.parameters():
        param_size = param.data.numel()
        param.data = flat_tensor[offset : offset + param_size].view_as(param.data)
        offset += param_size


def shard_tensor(tensor: torch.Tensor, num_shards: int) -> List[torch.Tensor]:
    """Split a 1D tensor into num_shards equal-sized shards.

    Args:
        tensor: 1D tensor to shard
        num_shards: Number of shards

    Returns:
        List of shards (may have unequal sizes if tensor size not divisible by num_shards)

    Raises:
        ValueError: If num_shards <= 0
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    if tensor.dim() != 1:
        raise ValueError("tensor must be 1D")

    shard_size = (tensor.numel() + num_shards - 1) // num_shards
    shards = []

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, tensor.numel())
        if start_idx < tensor.numel():
            shards.append(tensor[start_idx:end_idx].clone())

    return shards


def merge_shards(shards: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of tensor shards back into a single tensor.

    Args:
        shards: List of tensor shards

    Returns:
        Merged 1D tensor
    """
    if not shards:
        raise ValueError("shards list cannot be empty")

    return torch.cat(shards)


def tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes.

    Args:
        tensor: PyTorch tensor

    Returns:
        Size in bytes
    """
    return tensor.element_size() * tensor.numel()


def compute_compression_ratio(
    original_size: int, compressed_size: int
) -> float:
    """Compute compression ratio as a fraction.

    Args:
        original_size: Original size in bytes
        compressed_size: Compressed size in bytes

    Returns:
        Compression ratio (compressed / original)
    """
    if original_size == 0:
        return 0.0
    return compressed_size / original_size


def reset_cold_start_tracking() -> None:
    """Reset cold start tracking for testing."""
    global _function_invocation_count
    _function_invocation_count = {}
