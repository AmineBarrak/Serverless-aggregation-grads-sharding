#!/usr/bin/env python3
"""
VGG-16 Shard Sweep on AWS Lambda.

Runs VGG-16 aggregation with M=1,2,4,8,16 shards on real Lambda,
measuring S3 read time, compute time, S3 write time, and cost.

For each M:
  1. Generates and uploads sharded gradients to S3
  2. Invokes Lambda (parallel for M>1) and collects timing
  3. Saves results

Prerequisites:
  - setup_aws.py has been run (S3 bucket, IAM role, Lambda function exist)
  - pip install boto3 numpy

Usage:
    python run_vgg16_shard_sweep.py --bucket grads-sharding-exp-961341528585
    python run_vgg16_shard_sweep.py --bucket grads-sharding-exp-961341528585 --shards 1 2 4
"""

import argparse
import json
import time
import io
import sys
import concurrent.futures
from datetime import datetime
from pathlib import Path

try:
    import boto3
    import numpy as np
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 and numpy required. Run: pip install boto3 numpy")
    sys.exit(1)

# VGG-16 config
MODEL_NAME = "vgg16"
NUM_PARAMS = 134_301_514   # 138M params, ~512.3 MB gradient
GRAD_MB = NUM_PARAMS * 4 / (1024 * 1024)
NUM_CLIENTS = 20
NUM_ROUNDS = 3
NUM_REPS = 5
FUNCTION_NAME = "grads-sharding-agg-vgg16"

# AWS Lambda pricing (us-east-1)
LAMBDA_PRICE_PER_GB_S = 0.0000166667
S3_PUT_PRICE_PER_1K = 0.005
S3_GET_PRICE_PER_1K = 0.0004


def upload_gradients(s3, bucket, num_shards):
    """Generate and upload VGG-16 gradients for a given shard count."""
    prefix = f"gradients/vgg16_m{num_shards}"

    # Check if already uploaded
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if resp.get('KeyCount', 0) > 0:
            print(f"  Gradients for M={num_shards} already in S3, skipping upload.")
            return
    except Exception:
        pass

    shard_size = NUM_PARAMS // num_shards if num_shards > 1 else NUM_PARAMS
    shard_mb = shard_size * 4 / (1024 * 1024)
    total_files = NUM_ROUNDS * NUM_CLIENTS * num_shards
    total_mb = NUM_ROUNDS * NUM_CLIENTS * GRAD_MB

    print(f"  Uploading: {total_files} files ({total_mb:.0f} MB total)")
    print(f"  Shard size: {shard_mb:.1f} MB x {num_shards} shards")

    t_start = time.time()
    count = 0

    for round_id in range(NUM_ROUNDS):
        for client_id in range(NUM_CLIENTS):
            if num_shards == 1:
                grad = np.random.randn(NUM_PARAMS).astype(np.float32)
                key = f"{prefix}/round_{round_id}/client_{client_id}.npy"
                buf = io.BytesIO()
                np.save(buf, grad)
                buf.seek(0)
                s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
                del grad, buf
            else:
                shard_params = NUM_PARAMS // num_shards
                for shard_idx in range(num_shards):
                    grad_shard = np.random.randn(shard_params).astype(np.float32)
                    key = f"{prefix}/round_{round_id}/client_{client_id}_shard_{shard_idx}.npy"
                    buf = io.BytesIO()
                    np.save(buf, grad_shard)
                    buf.seek(0)
                    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
                    del grad_shard, buf

            count += 1
            if count % 5 == 0:
                elapsed = time.time() - t_start
                print(f"\r    Progress: {count}/{NUM_ROUNDS * NUM_CLIENTS} clients "
                      f"({elapsed:.0f}s)", end='', flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Upload done in {elapsed:.0f}s")


def invoke_lambda(client, func_name, payload, max_retries=5):
    """Invoke Lambda with retry logic."""
    for attempt in range(max_retries):
        try:
            t_start = time.perf_counter()
            response = client.invoke(
                FunctionName=func_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload),
            )
            t_end = time.perf_counter()

            response_payload = json.loads(response['Payload'].read().decode())

            if 'FunctionError' in response:
                print(f"\n    ERROR: {response_payload}")
                return None

            return {
                'response': response_payload,
                'wall_clock_ms': (t_end - t_start) * 1000,
            }
        except Exception as e:
            if 'TooManyRequests' in str(e) or 'Rate Exceeded' in str(e):
                wait = 2 ** attempt
                print(f" [rate limited, retry in {wait}s]", end='', flush=True)
                time.sleep(wait)
            else:
                raise
    return None


def run_shard_experiment(lambda_client, bucket, num_shards):
    """Run experiment for VGG-16 with a specific shard count."""
    prefix = f"gradients/vgg16_m{num_shards}"
    # For results, write to a different prefix per M
    memory_mb = 3008  # VGG-16 always uses max Lambda memory

    print(f"\n{'='*70}")
    print(f"  VGG-16 Shard Sweep: M={num_shards}")
    print(f"  Gradient: {GRAD_MB:.1f} MB | Per shard: {GRAD_MB/num_shards:.1f} MB")
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS} | Reps: {NUM_REPS}")
    print(f"{'='*70}")

    all_results = []

    for round_id in range(NUM_ROUNDS):
        for rep in range(NUM_REPS):
            label = f"round={round_id}, rep={rep}"

            if num_shards == 1:
                # Single invocation
                payload = {
                    'bucket': bucket,
                    'model_name': f'vgg16_m{num_shards}',
                    'shard_idx': 0,
                    'num_shards': 1,
                    'num_clients': NUM_CLIENTS,
                    'round_id': round_id,
                }
                result = invoke_lambda(lambda_client, FUNCTION_NAME, payload)
                if result is None:
                    print(f"  [{label}] FAILED")
                    continue

                resp = result['response']
                timings = resp['timings']
                wall_ms = result['wall_clock_ms']
                agg_ms = timings['aggregation_ms']
                s3_read_ms = timings['s3_read_total_ms']
                compute_ms = timings['compute_ms']
                s3_write_ms = timings['s3_write_ms']

                print(f"  [{label}] wall={wall_ms:.0f}ms | agg={agg_ms:.0f}ms "
                      f"(S3read={s3_read_ms:.0f} + compute={compute_ms:.0f} + S3write={s3_write_ms:.0f})")

                all_results.append({
                    'round_id': round_id, 'rep': rep,
                    'wall_clock_ms': wall_ms,
                    'aggregation_ms': agg_ms,
                    's3_read_ms': s3_read_ms,
                    'compute_ms': compute_ms,
                    's3_write_ms': s3_write_ms,
                    'cold_start': rep == 0 and round_id == 0,
                })

            else:
                # Parallel shard invocation
                def invoke_shard(shard_idx):
                    payload = {
                        'bucket': bucket,
                        'model_name': f'vgg16_m{num_shards}',
                        'shard_idx': shard_idx,
                        'num_shards': num_shards,
                        'num_clients': NUM_CLIENTS,
                        'round_id': round_id,
                    }
                    return invoke_lambda(lambda_client, FUNCTION_NAME, payload)

                t_par_start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as executor:
                    futures = {executor.submit(invoke_shard, s): s for s in range(num_shards)}
                    shard_results = {}
                    for future in concurrent.futures.as_completed(futures):
                        s_idx = futures[future]
                        shard_results[s_idx] = future.result()
                t_par_end = time.perf_counter()

                if any(r is None for r in shard_results.values()):
                    print(f"  [{label}] FAILED (some shards errored)")
                    continue

                parallel_wall_ms = (t_par_end - t_par_start) * 1000
                shard_agg_times = []
                shard_s3_read_times = []
                shard_compute_times = []
                shard_s3_write_times = []

                for s_idx in range(num_shards):
                    t = shard_results[s_idx]['response']['timings']
                    shard_agg_times.append(t['aggregation_ms'])
                    shard_s3_read_times.append(t['s3_read_total_ms'])
                    shard_compute_times.append(t['compute_ms'])
                    shard_s3_write_times.append(t['s3_write_ms'])

                max_agg = max(shard_agg_times)
                max_s3_read = max(shard_s3_read_times)
                max_compute = max(shard_compute_times)

                print(f"  [{label}] wall={parallel_wall_ms:.0f}ms | "
                      f"max_agg={max_agg:.0f}ms | "
                      f"shard_aggs=[{', '.join(f'{t:.0f}' for t in shard_agg_times)}]")

                all_results.append({
                    'round_id': round_id, 'rep': rep,
                    'wall_clock_ms': parallel_wall_ms,
                    'max_shard_agg_ms': max_agg,
                    'shard_agg_times_ms': shard_agg_times,
                    'shard_s3_read_times_ms': shard_s3_read_times,
                    'shard_compute_times_ms': shard_compute_times,
                    'shard_s3_write_times_ms': shard_s3_write_times,
                    'cold_start': rep == 0 and round_id == 0,
                })

    if not all_results:
        print("  No successful results!")
        return None

    # Exclude cold start
    warm = [r for r in all_results if not r.get('cold_start', False)]
    if not warm:
        warm = all_results

    if num_shards == 1:
        agg_times = [r['aggregation_ms'] for r in warm]
        wall_times = [r['wall_clock_ms'] for r in warm]
        s3_read_times = [r['s3_read_ms'] for r in warm]
        compute_times = [r['compute_ms'] for r in warm]
        s3_write_times = [r['s3_write_ms'] for r in warm]
    else:
        agg_times = [r['max_shard_agg_ms'] for r in warm]
        wall_times = [r['wall_clock_ms'] for r in warm]
        s3_read_times = [max(r['shard_s3_read_times_ms']) for r in warm]
        compute_times = [max(r['shard_compute_times_ms']) for r in warm]
        s3_write_times = [max(r['shard_s3_write_times_ms']) for r in warm]

    # Compute cost
    memory_gb = memory_mb / 1024
    mean_agg_s = np.mean(agg_times) / 1000
    lambda_cost = memory_gb * mean_agg_s * LAMBDA_PRICE_PER_GB_S * num_shards
    num_gets = NUM_CLIENTS * num_shards
    num_puts = num_shards
    s3_cost = (num_gets / 1000) * S3_GET_PRICE_PER_1K + (num_puts / 1000) * S3_PUT_PRICE_PER_1K
    total_cost = lambda_cost + s3_cost

    # S3 operations per round
    s3_ops = NUM_CLIENTS * num_shards + num_shards  # GETs + PUTs

    summary = {
        'model': 'VGG-16',
        'gradient_mb': GRAD_MB,
        'num_shards': num_shards,
        'shard_mb': GRAD_MB / num_shards,
        'memory_mb': memory_mb,
        'num_clients': NUM_CLIENTS,
        'num_warm_invocations': len(warm),
        'wall_clock_ms': {'mean': float(np.mean(wall_times)), 'std': float(np.std(wall_times))},
        'aggregation_ms': {'mean': float(np.mean(agg_times)), 'std': float(np.std(agg_times))},
        's3_read_ms': {'mean': float(np.mean(s3_read_times)), 'std': float(np.std(s3_read_times))},
        'compute_ms': {'mean': float(np.mean(compute_times)), 'std': float(np.std(compute_times))},
        's3_write_ms': {'mean': float(np.mean(s3_write_times)), 'std': float(np.std(s3_write_times))},
        's3_ops_per_round': s3_ops,
        'cost_per_round': {
            'lambda_compute': lambda_cost,
            's3_io': s3_cost,
            'total': total_cost,
        },
        'cost_per_1k_rounds': total_cost * 1000,
        'cold_start_ms': all_results[0]['wall_clock_ms'] if all_results else None,
        's3_throughput_mbps': float(
            (NUM_CLIENTS * GRAD_MB / num_shards) / (np.mean(s3_read_times) / 1000)
        ),
        'all_results': all_results,
    }

    print(f"\n  --- Summary (M={num_shards}, warm) ---")
    print(f"  Wall-clock:    {np.mean(wall_times):.0f} +/- {np.std(wall_times):.0f} ms")
    print(f"  Aggregation:   {np.mean(agg_times):.0f} +/- {np.std(agg_times):.0f} ms")
    print(f"  S3 reads:      {np.mean(s3_read_times):.0f} +/- {np.std(s3_read_times):.0f} ms")
    print(f"  Compute:       {np.mean(compute_times):.0f} +/- {np.std(compute_times):.0f} ms")
    print(f"  S3 write:      {np.mean(s3_write_times):.0f} +/- {np.std(s3_write_times):.0f} ms")
    print(f"  S3 throughput: {summary['s3_throughput_mbps']:.1f} MB/s per function")
    print(f"  S3 ops/round:  {s3_ops}")
    print(f"  Lambda cost:   ${lambda_cost:.6f}/round")
    print(f"  S3 cost:       ${s3_cost:.6f}/round")
    print(f"  Total cost:    ${total_cost:.6f}/round  (${total_cost*1000:.4f}/1K rounds)")
    print(f"  Cold start:    {summary['cold_start_ms']:.0f} ms")

    return summary


def main():
    parser = argparse.ArgumentParser(description="VGG-16 Shard Sweep on AWS Lambda")
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--shards', nargs='+', type=int, default=[1, 2, 4, 8, 16],
                        help='Shard counts to test (default: 1 2 4 8 16)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip gradient upload (use existing data)')
    parser.add_argument('--output', default='results/vgg16_shard_sweep',
                        help='Output directory')
    args = parser.parse_args()

    boto_config = Config(read_timeout=900, connect_timeout=10, retries={'max_attempts': 3})
    lambda_client = boto3.client('lambda', region_name=args.region, config=boto_config)
    s3_client = boto3.client('s3', region_name=args.region)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  VGG-16 SHARD SWEEP ON AWS LAMBDA")
    print(f"  Bucket: {args.bucket}")
    print(f"  Shard counts: {args.shards}")
    print(f"  {NUM_CLIENTS} clients x {NUM_ROUNDS} rounds x {NUM_REPS} reps")
    print(f"{'#'*70}")

    all_summaries = []

    for M in args.shards:
        # Step 1: Upload gradients
        if not args.skip_upload:
            print(f"\n--- Uploading VGG-16 gradients (M={M}) ---")
            upload_gradients(s3_client, args.bucket, M)

        # Step 2: Run experiment
        summary = run_shard_experiment(lambda_client, args.bucket, M)
        if summary:
            all_summaries.append(summary)
            out_file = output_dir / f"vgg16_m{M}.json"
            with open(out_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  Saved: {out_file}")

        # Brief pause between shard counts to avoid throttling
        if M != args.shards[-1]:
            print("\n  Pausing 10s before next shard count...")
            time.sleep(10)

    # Save combined results
    combined = {
        'experiment': 'VGG-16 Shard Sweep on AWS Lambda',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'bucket': args.bucket, 'region': args.region,
            'model': 'VGG-16', 'gradient_mb': GRAD_MB,
            'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
            'num_reps': NUM_REPS, 'shard_counts': args.shards,
        },
        'results': all_summaries,
    }
    combined_file = output_dir / 'vgg16_shard_sweep_results.json'
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    # Print comparison table
    print(f"\n\n{'='*90}")
    print(f"  VGG-16 SHARD SWEEP - FINAL COMPARISON")
    print(f"{'='*90}")
    print(f"{'M':>3} {'Shard(MB)':>9} {'Agg(ms)':>9} {'S3read(ms)':>11} {'Compute(ms)':>11} "
          f"{'S3write(ms)':>11} {'S3ops':>6} {'S3 MB/s':>8} {'Lambda$':>9} {'S3$':>9} {'Total$':>9}")
    print(f"{'-'*90}")
    for s in all_summaries:
        M = s['num_shards']
        print(f"{M:>3} {s['shard_mb']:>9.1f} "
              f"{s['aggregation_ms']['mean']:>9.0f} "
              f"{s['s3_read_ms']['mean']:>11.0f} "
              f"{s['compute_ms']['mean']:>11.0f} "
              f"{s['s3_write_ms']['mean']:>11.0f} "
              f"{s['s3_ops_per_round']:>6} "
              f"{s['s3_throughput_mbps']:>8.1f} "
              f"${s['cost_per_round']['lambda_compute']:>8.6f} "
              f"${s['cost_per_round']['s3_io']:>8.6f} "
              f"${s['cost_per_round']['total']:>8.6f}")
    print(f"{'='*90}")
    print(f"\nResults saved to: {combined_file}")


if __name__ == '__main__':
    main()
