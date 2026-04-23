#!/usr/bin/env python3
"""
Generate synthetic gradient vectors and upload to S3 for RQ3 experiments.

For each model, generates N client gradient vectors (random float32)
and uploads them to S3 in .npy format under the rq3/ prefix.

Uploads TWO formats for each model:
  1. Full gradients:   rq3/{model}/round_{r}/client_{c}.npy
     (used by λ-FL and LIFL, which process full gradient vectors)
  2. Sharded gradients: rq3/{model}/round_{r}/client_{c}_shard_{s}.npy
     (used by GradsSharding, which splits into M shards)

For models that exceed Lambda's 3008 MB memory limit with full gradients
(i.e., 2 × grad_size > 3008 MB), only sharded gradients are uploaded
since λ-FL/LIFL cannot run on those models.

Usage:
    python generate_rq3_gradients.py --bucket grads-sharding-exp-961341528585
    python generate_rq3_gradients.py --bucket grads-sharding-exp-961341528585 --model resnet18
    python generate_rq3_gradients.py --bucket grads-sharding-exp-961341528585 --dry-run
"""

import argparse
import io
import time
import sys
import math
import numpy as np

try:
    import boto3
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# Model configurations: name -> num_parameters
# Memory requirement for streaming: 2 × grad_size (one for running_sum, one for current read)
MODELS = {
    'resnet18': {
        'params': 11_181_642,     # ~42.7 MB gradient
        'num_shards': 4,          # GradsSharding M=4
    },
    'vgg16': {
        'params': 134_301_514,    # ~512.3 MB gradient
        'num_shards': 4,          # GradsSharding M=4
    },
    'gpt2_large': {
        'params': 774_030_080,    # ~2952.7 MB gradient
        'num_shards': 4,          # GradsSharding M=4
    },
    'synthetic_5gb': {
        'params': 1_342_177_280,  # ~5120 MB gradient (5 GB)
        'num_shards': 8,          # GradsSharding M=8 (each shard ~640 MB)
    },
}

NUM_CLIENTS = 20
NUM_ROUNDS = 3

# Lambda memory limit
LAMBDA_MAX_MEMORY_MB = 3008


def grad_size_mb(num_params):
    """Gradient size in MB for float32 parameters."""
    return num_params * 4 / (1024 * 1024)


def streaming_memory_mb(size_mb):
    """Memory needed for streaming accumulation: 2 × input_size."""
    return 2 * size_mb


def can_run_full_gradient(num_params):
    """Check if λ-FL/LIFL can process this model's full gradient on Lambda."""
    mem_needed = streaming_memory_mb(grad_size_mb(num_params))
    return mem_needed <= LAMBDA_MAX_MEMORY_MB


def upload_npy(s3, bucket, key, arr, dry_run=False):
    """Serialize numpy array and upload to S3. Returns bytes uploaded."""
    buf = io.BytesIO()
    np.save(buf, arr)
    nbytes = buf.tell()
    if not dry_run:
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return nbytes


def generate_for_model(s3, bucket, model_name, config, dry_run=False):
    """Generate and upload gradients for one model."""
    num_params = config['params']
    num_shards = config['num_shards']
    gsize = grad_size_mb(num_params)
    full_feasible = can_run_full_gradient(num_params)

    print(f"\n{'='*65}")
    print(f"  Model: {model_name}")
    print(f"  Parameters:     {num_params:,}")
    print(f"  Gradient size:  {gsize:.1f} MB")
    print(f"  Streaming mem:  {streaming_memory_mb(gsize):.0f} MB (2 × {gsize:.0f})")
    print(f"  Lambda limit:   {LAMBDA_MAX_MEMORY_MB} MB")
    print(f"  Full-gradient feasible (λ-FL/LIFL): {'YES' if full_feasible else 'NO — exceeds Lambda memory'}")
    print(f"  GradsSharding shards (M): {num_shards} → {gsize/num_shards:.0f} MB/shard → streaming: {2*gsize/num_shards:.0f} MB")
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}")
    if dry_run:
        print(f"  *** DRY RUN — no uploads ***")

    bytes_uploaded = 0
    files_uploaded = 0
    t_start = time.time()

    for round_id in range(NUM_ROUNDS):
        for client_id in range(NUM_CLIENTS):
            # --- Full gradient (for λ-FL / LIFL) ---
            if full_feasible:
                grad = np.random.randn(num_params).astype(np.float32)
                key = f"rq3/{model_name}/round_{round_id}/client_{client_id}.npy"
                nbytes = upload_npy(s3, bucket, key, grad, dry_run)
                bytes_uploaded += nbytes
                files_uploaded += 1
                del grad

            # --- Sharded gradients (for GradsSharding) ---
            shard_size = num_params // num_shards
            for shard_idx in range(num_shards):
                shard = np.random.randn(shard_size).astype(np.float32)
                key = f"rq3/{model_name}/round_{round_id}/client_{client_id}_shard_{shard_idx}.npy"
                nbytes = upload_npy(s3, bucket, key, shard, dry_run)
                bytes_uploaded += nbytes
                files_uploaded += 1
                del shard

            # Progress
            done = round_id * NUM_CLIENTS + client_id + 1
            total = NUM_ROUNDS * NUM_CLIENTS
            elapsed = time.time() - t_start
            rate = (bytes_uploaded / (1024**2)) / elapsed if elapsed > 0 else 0
            print(f"\r  Progress: {done}/{total} clients | "
                  f"{files_uploaded} files | "
                  f"{bytes_uploaded/(1024**2):.0f} MB | "
                  f"{rate:.1f} MB/s", end='', flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Done: {files_uploaded} files, {bytes_uploaded/(1024**3):.2f} GB in {elapsed:.0f}s")

    return {
        'model': model_name,
        'files': files_uploaded,
        'bytes': bytes_uploaded,
        'elapsed_s': elapsed,
        'full_gradient_uploaded': full_feasible,
    }


def print_plan(models_to_run):
    """Print upload plan before executing."""
    print(f"\n{'='*65}")
    print(f"  RQ3 GRADIENT UPLOAD PLAN")
    print(f"{'='*65}")

    total_files = 0
    total_gb = 0

    for name, cfg in models_to_run.items():
        num_params = cfg['params']
        num_shards = cfg['num_shards']
        gsize = grad_size_mb(num_params)
        full_ok = can_run_full_gradient(num_params)

        # Files per client per round
        files_per_client = num_shards  # sharded always
        if full_ok:
            files_per_client += 1  # plus full gradient

        n_files = NUM_ROUNDS * NUM_CLIENTS * files_per_client

        # Bytes per client per round
        bytes_per_client = num_params * 4  # sharded = same total bytes
        if full_ok:
            bytes_per_client += num_params * 4  # full gradient copy

        n_bytes = NUM_ROUNDS * NUM_CLIENTS * bytes_per_client

        total_files += n_files
        total_gb += n_bytes / (1024**3)

        print(f"  {name:<18} {gsize:>8.1f} MB × {NUM_CLIENTS} clients × {NUM_ROUNDS} rounds")
        print(f"    {'Full grad:':>14} {'YES' if full_ok else 'SKIP (exceeds Lambda 3008MB)'}")
        print(f"    {'Sharded:':>14} M={num_shards} ({gsize/num_shards:.0f} MB/shard)")
        print(f"    {'Files:':>14} {n_files:,}")
        print(f"    {'Size:':>14} {n_bytes/(1024**3):.2f} GB")

    print(f"\n  TOTAL: {total_files:,} files, {total_gb:.1f} GB")
    print(f"  Estimated S3 PUT cost: ${total_files * 0.005 / 1000:.2f}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Generate RQ3 synthetic gradients for S3")
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--model', default=None, choices=list(MODELS.keys()),
                        help='Single model (default: all)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without uploading')
    args = parser.parse_args()

    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS

    # Show plan
    print_plan(models_to_run)

    if args.dry_run:
        print("\n  DRY RUN complete. No files uploaded.")
        return

    # Confirm
    resp = input("\n  Proceed with upload? [y/N] ")
    if resp.lower() != 'y':
        print("  Aborted.")
        return

    s3 = boto3.client('s3', region_name=args.region)

    # Verify bucket
    try:
        s3.head_bucket(Bucket=args.bucket)
    except Exception as e:
        print(f"ERROR: Cannot access bucket '{args.bucket}': {e}")
        sys.exit(1)

    results = []
    t_global = time.time()

    for model_name, config in models_to_run.items():
        r = generate_for_model(s3, args.bucket, model_name, config)
        results.append(r)

    elapsed = time.time() - t_global
    total_bytes = sum(r['bytes'] for r in results)
    total_files = sum(r['files'] for r in results)

    print(f"\n{'='*65}")
    print(f"  ALL DONE")
    print(f"  {total_files:,} files | {total_bytes/(1024**3):.2f} GB | {elapsed:.0f}s")
    print(f"  Estimated S3 PUT cost: ${total_files * 0.005 / 1000:.4f}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
