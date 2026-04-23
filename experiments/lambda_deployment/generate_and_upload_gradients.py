#!/usr/bin/env python3
"""
Generate synthetic gradient vectors and upload to S3.

For each model, generates N client gradient vectors (random float32)
and uploads them to S3 in .npy format. For sharded models (GPT-2),
also generates M shards per client.

Usage:
    python generate_and_upload_gradients.py --bucket grads-sharding-exp
    python generate_and_upload_gradients.py --bucket grads-sharding-exp --model resnet18
"""

import argparse
import io
import time
import sys
import numpy as np

try:
    import boto3
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# Model configurations: name -> num_parameters
MODELS = {
    'resnet18':    11_181_642,    # 11.7M params, ~42.7 MB
    'vgg16':       134_301_514,   # 138M params,  ~512.3 MB
    'gpt2_medium': 354_823_168,   # 345M params,  ~1,353 MB
    'gpt2_large':  774_030_080,   # 774M params,  ~2,953 MB
}

# Sharding config: full model for small, M=4 shards for large
SHARD_CONFIG = {
    'resnet18':    1,   # no sharding needed
    'vgg16':       1,   # fits in 10GB Lambda with streaming
    'gpt2_medium': 4,   # sharded: ~338 MB per shard
    'gpt2_large':  4,   # sharded: ~738 MB per shard
}

NUM_CLIENTS = 20
NUM_ROUNDS = 3  # multiple rounds for statistical significance


def upload_gradient(s3, bucket, key, grad_array):
    """Serialize numpy array and upload to S3."""
    buf = io.BytesIO()
    np.save(buf, grad_array)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return buf.tell()


def generate_model_gradients(s3, bucket, model_name, num_params, num_shards):
    """Generate and upload gradients for one model across all rounds."""
    grad_mb = num_params * 4 / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"  Parameters:  {num_params:,}")
    print(f"  Gradient:    {grad_mb:.1f} MB")
    print(f"  Shards (M):  {num_shards}")
    if num_shards > 1:
        shard_params = num_params // num_shards
        print(f"  Per shard:   {shard_params:,} params ({shard_params * 4 / (1024*1024):.1f} MB)")
    print(f"  Clients (N): {NUM_CLIENTS}")
    print(f"  Rounds:      {NUM_ROUNDS}")
    total_uploads = NUM_ROUNDS * NUM_CLIENTS * num_shards
    total_mb = NUM_ROUNDS * NUM_CLIENTS * grad_mb
    print(f"  Total uploads: {total_uploads} files ({total_mb:.0f} MB)")
    print(f"{'='*60}")

    bytes_uploaded = 0
    t_start = time.time()

    for round_id in range(NUM_ROUNDS):
        for client_id in range(NUM_CLIENTS):
            if num_shards == 1:
                # Full gradient
                grad = np.random.randn(num_params).astype(np.float32)
                key = f"gradients/{model_name}/round_{round_id}/client_{client_id}.npy"
                size = upload_gradient(s3, bucket, key, grad)
                bytes_uploaded += size
            else:
                # Sharded gradient
                shard_size = num_params // num_shards
                for shard_idx in range(num_shards):
                    grad_shard = np.random.randn(shard_size).astype(np.float32)
                    key = f"gradients/{model_name}/round_{round_id}/client_{client_id}_shard_{shard_idx}.npy"
                    size = upload_gradient(s3, bucket, key, grad_shard)
                    bytes_uploaded += size

            # Progress
            done = round_id * NUM_CLIENTS + client_id + 1
            total = NUM_ROUNDS * NUM_CLIENTS
            elapsed = time.time() - t_start
            rate_mbs = (bytes_uploaded / (1024*1024)) / elapsed if elapsed > 0 else 0
            print(f"\r  Uploading: {done}/{total} clients "
                  f"({bytes_uploaded/(1024*1024):.0f} MB, {rate_mbs:.1f} MB/s)", end='', flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s ({bytes_uploaded/(1024*1024):.0f} MB uploaded)")
    return bytes_uploaded


def main():
    parser = argparse.ArgumentParser(description="Generate and upload synthetic gradients to S3")
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--model', default=None,
                        help=f'Single model to upload ({", ".join(MODELS.keys())}). Default: all')
    parser.add_argument('--shards', type=int, default=None,
                        help='Override shard count M (e.g., --model vgg16 --shards 4)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()

    s3 = boto3.client('s3', region_name=args.region)

    # Verify bucket exists
    try:
        s3.head_bucket(Bucket=args.bucket)
        print(f"Using bucket: {args.bucket}")
    except Exception as e:
        print(f"ERROR: Cannot access bucket '{args.bucket}': {e}")
        print("Create it first with: aws s3 mb s3://{args.bucket} --region {args.region}")
        sys.exit(1)

    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    if args.model and args.model not in MODELS:
        print(f"ERROR: Unknown model '{args.model}'. Choose from: {', '.join(MODELS.keys())}")
        sys.exit(1)

    total_bytes = 0
    t_global = time.time()

    for model_name, num_params in models_to_run.items():
        num_shards = args.shards if args.shards else SHARD_CONFIG[model_name]
        total_bytes += generate_model_gradients(s3, args.bucket, model_name, num_params, num_shards)

    elapsed = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"ALL DONE: {total_bytes/(1024**3):.2f} GB uploaded in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
