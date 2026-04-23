"""
Lambda function for RQ1 serverless aggregation validation.

This function:
  1. Reads N client gradient shards from S3
  2. Performs FedAvg aggregation (streaming accumulation)
  3. Writes the averaged result back to S3
  4. Returns detailed timing breakdown

Deployed with numpy layer for efficient tensor operations.
"""

import json
import time
import os
import io
import boto3
import numpy as np

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Event format:
    {
        "bucket": "grads-sharding-exp",
        "model_name": "resnet18",
        "shard_idx": 0,          # which shard (0 for full-model, 0..M-1 for sharded)
        "num_shards": 1,         # M (1 = full model aggregation)
        "num_clients": 20,       # N
        "round_id": 0
    }
    """
    bucket = event['bucket']
    model_name = event['model_name']
    shard_idx = event.get('shard_idx', 0)
    num_shards = event.get('num_shards', 1)
    num_clients = event['num_clients']
    round_id = event.get('round_id', 0)

    timings = {}
    t_total_start = time.perf_counter()

    # --- Phase 1: Streaming aggregation from S3 ---
    t_agg_start = time.perf_counter()
    running_sum = None
    s3_read_times = []
    s3_bytes_total = 0

    for client_id in range(num_clients):
        if num_shards == 1:
            key = f"gradients/{model_name}/round_{round_id}/client_{client_id}.npy"
        else:
            key = f"gradients/{model_name}/round_{round_id}/client_{client_id}_shard_{shard_idx}.npy"

        # Time S3 read
        t_read_start = time.perf_counter()
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read()
        s3_bytes_total += len(body)
        grad = np.load(io.BytesIO(body))
        del body  # free raw bytes immediately
        t_read_end = time.perf_counter()
        s3_read_times.append(t_read_end - t_read_start)

        # Streaming accumulation: add to running sum (float32 to save memory)
        if running_sum is None:
            running_sum = grad.copy()
        else:
            np.add(running_sum, grad, out=running_sum)
        del grad  # free immediately

    # Divide by N to get average (in-place)
    result = np.divide(running_sum, num_clients, out=running_sum)
    t_agg_end = time.perf_counter()

    timings['aggregation_ms'] = (t_agg_end - t_agg_start) * 1000
    timings['s3_read_total_ms'] = sum(s3_read_times) * 1000
    timings['s3_read_mean_ms'] = (sum(s3_read_times) / len(s3_read_times)) * 1000
    timings['s3_bytes_read'] = s3_bytes_total
    timings['compute_ms'] = timings['aggregation_ms'] - timings['s3_read_total_ms']

    # --- Phase 2: Write result back to S3 ---
    t_write_start = time.perf_counter()
    buf = io.BytesIO()
    np.save(buf, result)
    buf.seek(0)
    if num_shards == 1:
        out_key = f"results/{model_name}/round_{round_id}/aggregated.npy"
    else:
        out_key = f"results/{model_name}/round_{round_id}/aggregated_shard_{shard_idx}.npy"
    s3.put_object(Bucket=bucket, Key=out_key, Body=buf.getvalue())
    t_write_end = time.perf_counter()
    timings['s3_write_ms'] = (t_write_end - t_write_start) * 1000

    t_total_end = time.perf_counter()
    timings['total_ms'] = (t_total_end - t_total_start) * 1000

    return {
        'statusCode': 200,
        'model_name': model_name,
        'shard_idx': shard_idx,
        'num_shards': num_shards,
        'num_clients': num_clients,
        'gradient_shape': list(result.shape),
        'gradient_size_mb': round(s3_bytes_total / num_clients / (1024 * 1024), 1),
        'total_gradient_read_mb': round(s3_bytes_total / (1024 * 1024), 1),
        'memory_limit_mb': int(context.memory_limit_in_mb),
        'memory_used_mb': round(result.nbytes * 2 / (1024 * 1024), 1),  # approx: result + running_sum
        'timings': timings,
        'remaining_time_ms': context.get_remaining_time_in_millis(),
    }
