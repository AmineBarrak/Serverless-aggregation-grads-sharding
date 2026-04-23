"""
Unified Lambda function for RQ3: Cross-Architecture Comparison.

Supports three aggregation modes:
  - grads-sharding: Each function aggregates one shard across all N clients
  - lambda-fl-leaf: Leaf aggregator reads k client gradients, computes partial sum
  - lambda-fl-root: Root aggregator reads leaf results, computes final average
  - lifl-level: Generic level aggregator (works for any level in LIFL's 3-level tree)

All modes use streaming accumulation: read one gradient at a time from S3,
accumulate into a running sum, divide at the end. Memory = 2 x input_size.

Deployed with numpy via AWSSDKPandas layer.
"""

import json
import time
import os
import io
import boto3
import numpy as np

s3 = boto3.client('s3')


def _read_npy_from_s3(bucket, key):
    """Read a numpy array from S3, return (array, read_time_s, bytes_read)."""
    t0 = time.perf_counter()
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read()
    nbytes = len(body)
    arr = np.load(io.BytesIO(body))
    del body
    elapsed = time.perf_counter() - t0
    return arr, elapsed, nbytes


def _write_npy_to_s3(bucket, key, arr):
    """Write a numpy array to S3, return write_time_s."""
    t0 = time.perf_counter()
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    elapsed = time.perf_counter() - t0
    return elapsed


def _streaming_aggregate(bucket, keys, num_inputs):
    """Stream-read N arrays from S3, accumulate sum, return (average, metrics)."""
    running_sum = None
    s3_read_times = []
    s3_bytes_total = 0

    for key in keys:
        arr, read_time, nbytes = _read_npy_from_s3(bucket, key)
        s3_read_times.append(read_time)
        s3_bytes_total += nbytes

        if running_sum is None:
            running_sum = arr.copy()
        else:
            np.add(running_sum, arr, out=running_sum)
        del arr

    result = np.divide(running_sum, num_inputs, out=running_sum)

    return result, {
        's3_read_total_s': sum(s3_read_times),
        's3_read_mean_s': sum(s3_read_times) / len(s3_read_times) if s3_read_times else 0,
        's3_bytes_read': s3_bytes_total,
        'num_inputs': len(keys),
    }


def _streaming_sum(bucket, keys):
    """Stream-read N arrays from S3, accumulate sum (no division). For leaf aggregators."""
    running_sum = None
    s3_read_times = []
    s3_bytes_total = 0
    count = 0

    for key in keys:
        arr, read_time, nbytes = _read_npy_from_s3(bucket, key)
        s3_read_times.append(read_time)
        s3_bytes_total += nbytes
        count += 1

        if running_sum is None:
            running_sum = arr.copy()
        else:
            np.add(running_sum, arr, out=running_sum)
        del arr

    return running_sum, count, {
        's3_read_total_s': sum(s3_read_times),
        's3_bytes_read': s3_bytes_total,
        'num_inputs': count,
    }


def handle_grads_sharding(event, context):
    """GradsSharding: aggregate one shard across all N clients."""
    bucket = event['bucket']
    model = event['model_name']
    shard_idx = event['shard_idx']
    num_shards = event['num_shards']
    num_clients = event['num_clients']
    round_id = event.get('round_id', 0)

    t_start = time.perf_counter()

    # Build S3 keys for this shard from all clients
    keys = [
        f"rq3/{model}/round_{round_id}/client_{c}_shard_{shard_idx}.npy"
        for c in range(num_clients)
    ]

    result, read_metrics = _streaming_aggregate(bucket, keys, num_clients)

    t_compute_done = time.perf_counter()

    # Write averaged shard back to S3
    out_key = f"rq3/{model}/round_{round_id}/agg_shard_{shard_idx}.npy"
    write_time = _write_npy_to_s3(bucket, out_key, result)

    t_end = time.perf_counter()

    return {
        'mode': 'grads-sharding',
        'shard_idx': shard_idx,
        'num_shards': num_shards,
        'gradient_shape': list(result.shape),
        'shard_size_mb': round(result.nbytes / (1024 * 1024), 1),
        'memory_limit_mb': int(context.memory_limit_in_mb),
        'timings': {
            'total_s': t_end - t_start,
            'aggregation_s': t_compute_done - t_start,
            's3_read_s': read_metrics['s3_read_total_s'],
            'compute_s': (t_compute_done - t_start) - read_metrics['s3_read_total_s'],
            's3_write_s': write_time,
            's3_bytes_read': read_metrics['s3_bytes_read'],
        },
    }


def handle_lambda_fl_leaf(event, context):
    """Lambda-FL leaf: read k client gradients, compute partial sum, write to S3."""
    bucket = event['bucket']
    model = event['model_name']
    leaf_id = event['leaf_id']
    client_ids = event['client_ids']  # list of client IDs assigned to this leaf
    round_id = event.get('round_id', 0)

    t_start = time.perf_counter()

    # Build S3 keys for assigned clients (full gradients)
    keys = [
        f"rq3/{model}/round_{round_id}/client_{c}.npy"
        for c in client_ids
    ]

    partial_sum, count, read_metrics = _streaming_sum(bucket, keys)

    t_compute_done = time.perf_counter()

    # Write partial sum to S3 (NOT averaged yet - root will divide by total N)
    out_key = f"rq3/{model}/round_{round_id}/leaf_{leaf_id}_sum.npy"
    write_time = _write_npy_to_s3(bucket, out_key, partial_sum)

    t_end = time.perf_counter()

    return {
        'mode': 'lambda-fl-leaf',
        'leaf_id': leaf_id,
        'num_clients': count,
        'gradient_shape': list(partial_sum.shape),
        'gradient_size_mb': round(partial_sum.nbytes / (1024 * 1024), 1),
        'memory_limit_mb': int(context.memory_limit_in_mb),
        'timings': {
            'total_s': t_end - t_start,
            'aggregation_s': t_compute_done - t_start,
            's3_read_s': read_metrics['s3_read_total_s'],
            'compute_s': (t_compute_done - t_start) - read_metrics['s3_read_total_s'],
            's3_write_s': write_time,
            's3_bytes_read': read_metrics['s3_bytes_read'],
        },
    }


def handle_lambda_fl_root(event, context):
    """Lambda-FL root: read leaf partial sums, compute final average."""
    bucket = event['bucket']
    model = event['model_name']
    num_leaves = event['num_leaves']
    num_clients = event['num_clients']  # total N for final division
    round_id = event.get('round_id', 0)

    t_start = time.perf_counter()

    # Read leaf partial sums
    keys = [
        f"rq3/{model}/round_{round_id}/leaf_{i}_sum.npy"
        for i in range(num_leaves)
    ]

    # Sum all leaf sums, then divide by total N
    running_sum = None
    s3_read_times = []
    s3_bytes_total = 0

    for key in keys:
        arr, read_time, nbytes = _read_npy_from_s3(bucket, key)
        s3_read_times.append(read_time)
        s3_bytes_total += nbytes
        if running_sum is None:
            running_sum = arr.copy()
        else:
            np.add(running_sum, arr, out=running_sum)
        del arr

    result = np.divide(running_sum, num_clients, out=running_sum)

    t_compute_done = time.perf_counter()

    # Write final aggregated gradient
    out_key = f"rq3/{model}/round_{round_id}/aggregated.npy"
    write_time = _write_npy_to_s3(bucket, out_key, result)

    t_end = time.perf_counter()

    return {
        'mode': 'lambda-fl-root',
        'num_leaves': num_leaves,
        'num_clients': num_clients,
        'gradient_shape': list(result.shape),
        'gradient_size_mb': round(result.nbytes / (1024 * 1024), 1),
        'memory_limit_mb': int(context.memory_limit_in_mb),
        'timings': {
            'total_s': t_end - t_start,
            'aggregation_s': t_compute_done - t_start,
            's3_read_s': sum(s3_read_times),
            'compute_s': (t_compute_done - t_start) - sum(s3_read_times),
            's3_write_s': write_time,
            's3_bytes_read': s3_bytes_total,
        },
    }


def handle_lifl_level(event, context):
    """LIFL level aggregator: read inputs from S3, average, write result.

    Works for any level in the hierarchy:
    - Level 1: reads client gradients, averages subset
    - Level 2: reads level-1 results, averages
    - Root: reads level-2 results, produces final
    """
    bucket = event['bucket']
    model = event['model_name']
    level = event['level']           # 1, 2, or 3 (root)
    agg_id = event['agg_id']         # aggregator ID within this level
    input_keys = event['input_keys'] # list of S3 keys to read
    num_inputs_total = event.get('num_inputs_total', len(event['input_keys']))
    round_id = event.get('round_id', 0)

    t_start = time.perf_counter()

    result, read_metrics = _streaming_aggregate(bucket, input_keys, len(input_keys))

    t_compute_done = time.perf_counter()

    # Write result
    out_key = f"rq3/{model}/round_{round_id}/lifl_L{level}_agg{agg_id}.npy"
    if level == 3:
        out_key = f"rq3/{model}/round_{round_id}/aggregated.npy"
    write_time = _write_npy_to_s3(bucket, out_key, result)

    t_end = time.perf_counter()

    return {
        'mode': f'lifl-level-{level}',
        'level': level,
        'agg_id': agg_id,
        'num_inputs': len(input_keys),
        'gradient_shape': list(result.shape),
        'gradient_size_mb': round(result.nbytes / (1024 * 1024), 1),
        'memory_limit_mb': int(context.memory_limit_in_mb),
        'output_key': out_key,
        'timings': {
            'total_s': t_end - t_start,
            'aggregation_s': t_compute_done - t_start,
            's3_read_s': read_metrics['s3_read_total_s'],
            'compute_s': (t_compute_done - t_start) - read_metrics['s3_read_total_s'],
            's3_write_s': write_time,
            's3_bytes_read': read_metrics['s3_bytes_read'],
        },
    }


def lambda_handler(event, context):
    """Unified entry point. Dispatches based on 'mode' field."""
    mode = event.get('mode', 'grads-sharding')

    if mode == 'grads-sharding':
        return handle_grads_sharding(event, context)
    elif mode == 'lambda-fl-leaf':
        return handle_lambda_fl_leaf(event, context)
    elif mode == 'lambda-fl-root':
        return handle_lambda_fl_root(event, context)
    elif mode == 'lifl-level':
        return handle_lifl_level(event, context)
    else:
        return {
            'statusCode': 400,
            'error': f'Unknown mode: {mode}. Use: grads-sharding, lambda-fl-leaf, lambda-fl-root, lifl-level'
        }
