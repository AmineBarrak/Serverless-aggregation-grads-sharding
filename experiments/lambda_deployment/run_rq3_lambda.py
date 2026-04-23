#!/usr/bin/env python3
"""
RQ3 Lambda Experiment: Cross-Architecture Comparison on Real AWS Lambda.

Runs all three FL aggregation architectures on real Lambda functions:
  1. GradsSharding — M parallel shard aggregators
  2. λ-FL          — two-level tree (√N leaves + 1 root)
  3. LIFL          — three-level hierarchy

For each model × architecture, measures:
  - End-to-end wall-clock latency
  - Lambda compute time (billed duration)
  - S3 I/O time
  - Pure compute time
  - Total cost (Lambda + S3)

When an architecture is INFEASIBLE (exceeds Lambda 3008 MB), it is
recorded as such rather than invoked.

Usage:
    python run_rq3_lambda.py --bucket grads-sharding-exp-961341528585
    python run_rq3_lambda.py --bucket grads-sharding-exp-961341528585 --model resnet18
    python run_rq3_lambda.py --bucket grads-sharding-exp-961341528585 --quick
"""

import argparse
import json
import math
import time
import sys
import os
import concurrent.futures
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────

NUM_CLIENTS = 20
NUM_ROUNDS = 3
NUM_REPS = 3        # repetitions per round for timing stability

# AWS pricing (us-east-1)
LAMBDA_PRICE_PER_GB_S = 0.0000166667
S3_PUT_PRICE_PER_1K = 0.005
S3_GET_PRICE_PER_1K = 0.0004

LAMBDA_MAX_MEMORY_MB = 3008

MODELS = {
    'resnet18': {
        'params': 11_181_642,
        'grad_mb': 42.7,
        'num_shards': 4,
    },
    'vgg16': {
        'params': 134_301_514,
        'grad_mb': 512.3,
        'num_shards': 4,
    },
    'gpt2_large': {
        'params': 774_030_080,
        'grad_mb': 2952.7,
        'num_shards': 4,
    },
    'synthetic_5gb': {
        'params': 1_342_177_280,
        'grad_mb': 5120.0,
        'num_shards': 8,
    },
}


def load_function_registry():
    """Load function registry created by setup_rq3_aws.py."""
    reg_path = os.path.join(os.path.dirname(__file__), "rq3_function_registry.json")
    if not os.path.exists(reg_path):
        print(f"ERROR: {reg_path} not found. Run setup_rq3_aws.py first.")
        sys.exit(1)
    with open(reg_path) as f:
        return json.load(f)


# ─── Lambda invocation helpers ───────────────────────────────────────────────

def invoke_lambda(client, func_name, payload, max_retries=5):
    """Invoke a Lambda function synchronously. Returns parsed response or None."""
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            response = client.invoke(
                FunctionName=func_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload),
            )
            wall_s = time.perf_counter() - t0

            resp_payload = json.loads(response['Payload'].read().decode())

            if 'FunctionError' in response:
                error_msg = resp_payload.get('errorMessage', str(resp_payload))
                print(f"\n    Lambda error: {error_msg[:200]}")
                return None

            return {
                'response': resp_payload,
                'wall_clock_s': wall_s,
            }
        except Exception as e:
            if 'TooManyRequestsException' in str(e) or 'Rate Exceeded' in str(e):
                wait = 2 ** attempt
                print(f" (rate-limited, retry in {wait}s)", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    return None


# ─── Architecture runners ────────────────────────────────────────────────────

def run_grads_sharding(lambda_client, registry, model_name, model_config,
                        bucket, round_id):
    """
    GradsSharding: invoke M shard aggregators in parallel.

    Each shard aggregator reads that shard from all N clients, averages, writes result.
    Wall-clock = max(shard times). Cost = sum(all shard costs).
    """
    func_info = registry[model_name]['grads_sharding']
    func_name = func_info['function_name']
    memory_mb = func_info['memory_mb']
    num_shards = model_config['num_shards']

    def invoke_shard(shard_idx):
        payload = {
            'mode': 'grads-sharding',
            'bucket': bucket,
            'model_name': model_name,
            'shard_idx': shard_idx,
            'num_shards': num_shards,
            'num_clients': NUM_CLIENTS,
            'round_id': round_id,
        }
        return invoke_lambda(lambda_client, func_name, payload)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = {executor.submit(invoke_shard, s): s for s in range(num_shards)}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            s_idx = futures[future]
            results[s_idx] = future.result()
    wall_s = time.perf_counter() - t0

    # Check for failures
    if any(r is None for r in results.values()):
        return None

    # Aggregate metrics across shards
    shard_timings = []
    total_s3_gets = NUM_CLIENTS * num_shards
    total_s3_puts = num_shards

    for s_idx in range(num_shards):
        t = results[s_idx]['response']['timings']
        shard_timings.append(t)

    # Lambda cost: each shard runs independently
    # Billed duration ≈ total_s per shard (Lambda bills by execution time)
    shard_durations = [t['total_s'] for t in shard_timings]
    lambda_cost = sum(
        (memory_mb / 1024) * d * LAMBDA_PRICE_PER_GB_S
        for d in shard_durations
    )
    s3_cost = (total_s3_gets / 1000) * S3_GET_PRICE_PER_1K + \
              (total_s3_puts / 1000) * S3_PUT_PRICE_PER_1K

    return {
        'architecture': 'grads_sharding',
        'wall_clock_s': wall_s,
        'max_shard_total_s': max(shard_durations),
        'shard_durations_s': shard_durations,
        's3_read_s': [t['s3_read_s'] for t in shard_timings],
        'compute_s': [t['compute_s'] for t in shard_timings],
        's3_write_s': [t['s3_write_s'] for t in shard_timings],
        's3_bytes_read': sum(t['s3_bytes_read'] for t in shard_timings),
        'num_lambda_invocations': num_shards,
        'num_s3_gets': total_s3_gets,
        'num_s3_puts': total_s3_puts,
        'memory_mb': memory_mb,
        'cost': {
            'lambda_compute': lambda_cost,
            's3_io': s3_cost,
            'total': lambda_cost + s3_cost,
        },
    }


def run_lambda_fl(lambda_client, registry, model_name, model_config,
                   bucket, round_id):
    """
    λ-FL: two-level tree aggregation.

    Level 1 (leaves): k = ⌈√N⌉ clients per leaf, ⌈N/k⌉ leaves in parallel.
                       Each leaf computes partial sum, writes to S3.
    Level 2 (root):   Reads all leaf sums, computes final average.

    Wall-clock = max(leaf times) + root time. Cost = sum(all invocations).
    """
    func_info = registry[model_name]['lambda_fl']
    func_name = func_info['function_name']
    memory_mb = func_info['memory_mb']

    # Topology: k clients per leaf, ceil(N/k) leaves
    k = math.ceil(math.sqrt(NUM_CLIENTS))  # clients per leaf
    num_leaves = math.ceil(NUM_CLIENTS / k)

    # Assign clients to leaves
    leaf_assignments = []
    for leaf_id in range(num_leaves):
        start = leaf_id * k
        end = min(start + k, NUM_CLIENTS)
        leaf_assignments.append(list(range(start, end)))

    # ── Phase 1: Leaf aggregators in parallel ──
    def invoke_leaf(leaf_id):
        payload = {
            'mode': 'lambda-fl-leaf',
            'bucket': bucket,
            'model_name': model_name,
            'leaf_id': leaf_id,
            'client_ids': leaf_assignments[leaf_id],
            'round_id': round_id,
        }
        return invoke_lambda(lambda_client, func_name, payload)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_leaves) as executor:
        futures = {executor.submit(invoke_leaf, i): i for i in range(num_leaves)}
        leaf_results = {}
        for future in concurrent.futures.as_completed(futures):
            lid = futures[future]
            leaf_results[lid] = future.result()
    t_leaves_done = time.perf_counter()

    if any(r is None for r in leaf_results.values()):
        return None

    # ── Phase 2: Root aggregator ──
    root_payload = {
        'mode': 'lambda-fl-root',
        'bucket': bucket,
        'model_name': model_name,
        'num_leaves': num_leaves,
        'num_clients': NUM_CLIENTS,
        'round_id': round_id,
    }
    root_result = invoke_lambda(lambda_client, func_name, root_payload)
    t_root_done = time.perf_counter()

    if root_result is None:
        return None

    wall_s = t_root_done - t0

    # Collect metrics
    leaf_timings = [leaf_results[i]['response']['timings'] for i in range(num_leaves)]
    root_timings = root_result['response']['timings']

    # S3 operations:
    # Leaves: N GETs (read client grads) + num_leaves PUTs (write partial sums)
    # Root: num_leaves GETs (read leaf sums) + 1 PUT (write final)
    total_s3_gets = NUM_CLIENTS + num_leaves
    total_s3_puts = num_leaves + 1
    total_invocations = num_leaves + 1

    # Cost: sum all invocations
    leaf_durations = [t['total_s'] for t in leaf_timings]
    root_duration = root_timings['total_s']
    all_durations = leaf_durations + [root_duration]

    lambda_cost = sum(
        (memory_mb / 1024) * d * LAMBDA_PRICE_PER_GB_S
        for d in all_durations
    )
    s3_cost = (total_s3_gets / 1000) * S3_GET_PRICE_PER_1K + \
              (total_s3_puts / 1000) * S3_PUT_PRICE_PER_1K

    return {
        'architecture': 'lambda_fl',
        'topology': {'k': k, 'num_leaves': num_leaves},
        'wall_clock_s': wall_s,
        'leaf_phase_s': t_leaves_done - t0,
        'root_phase_s': t_root_done - t_leaves_done,
        'max_leaf_total_s': max(leaf_durations),
        'leaf_durations_s': leaf_durations,
        'root_duration_s': root_duration,
        's3_read_s_leaves': [t['s3_read_s'] for t in leaf_timings],
        'compute_s_leaves': [t['compute_s'] for t in leaf_timings],
        's3_read_s_root': root_timings['s3_read_s'],
        'compute_s_root': root_timings['compute_s'],
        's3_bytes_read': sum(t['s3_bytes_read'] for t in leaf_timings) + root_timings['s3_bytes_read'],
        'num_lambda_invocations': total_invocations,
        'num_s3_gets': total_s3_gets,
        'num_s3_puts': total_s3_puts,
        'memory_mb': memory_mb,
        'cost': {
            'lambda_compute': lambda_cost,
            's3_io': s3_cost,
            'total': lambda_cost + s3_cost,
        },
    }


def run_lifl(lambda_client, registry, model_name, model_config,
              bucket, round_id):
    """
    LIFL: three-level hierarchical aggregation.

    Level 1: ∛N groups of ∛N clients each → ∛N aggregators (parallel)
    Level 2: Groups of ∛N level-1 results → fewer aggregators (parallel)
    Level 3: Root aggregates all level-2 results

    Wall-clock = max(L1) + max(L2) + L3. Cost = sum(all invocations).
    """
    func_info = registry[model_name]['lifl']
    func_name = func_info['function_name']
    memory_mb = func_info['memory_mb']

    # Topology: cube-root branching
    branch = max(2, math.ceil(NUM_CLIENTS ** (1/3)))

    # Level 1: group clients into clusters of `branch`
    l1_groups = []
    for i in range(0, NUM_CLIENTS, branch):
        group = list(range(i, min(i + branch, NUM_CLIENTS)))
        l1_groups.append(group)
    num_l1 = len(l1_groups)

    # ── Level 1: parallel ──
    def invoke_l1(agg_id):
        input_keys = [
            f"rq3/{model_name}/round_{round_id}/client_{c}.npy"
            for c in l1_groups[agg_id]
        ]
        payload = {
            'mode': 'lifl-level',
            'bucket': bucket,
            'model_name': model_name,
            'level': 1,
            'agg_id': agg_id,
            'input_keys': input_keys,
            'num_inputs_total': len(input_keys),
            'round_id': round_id,
        }
        return invoke_lambda(lambda_client, func_name, payload)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_l1) as executor:
        futures = {executor.submit(invoke_l1, i): i for i in range(num_l1)}
        l1_results = {}
        for future in concurrent.futures.as_completed(futures):
            aid = futures[future]
            l1_results[aid] = future.result()
    t_l1_done = time.perf_counter()

    if any(r is None for r in l1_results.values()):
        return None

    # ── Level 2: group L1 outputs ──
    l1_output_keys = [
        l1_results[i]['response']['output_key']
        for i in range(num_l1)
    ]

    l2_groups = []
    for i in range(0, num_l1, branch):
        l2_groups.append(l1_output_keys[i:i+branch])
    num_l2 = len(l2_groups)

    def invoke_l2(agg_id):
        payload = {
            'mode': 'lifl-level',
            'bucket': bucket,
            'model_name': model_name,
            'level': 2,
            'agg_id': agg_id,
            'input_keys': l2_groups[agg_id],
            'num_inputs_total': len(l2_groups[agg_id]),
            'round_id': round_id,
        }
        return invoke_lambda(lambda_client, func_name, payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_l2) as executor:
        futures = {executor.submit(invoke_l2, i): i for i in range(num_l2)}
        l2_results = {}
        for future in concurrent.futures.as_completed(futures):
            aid = futures[future]
            l2_results[aid] = future.result()
    t_l2_done = time.perf_counter()

    if any(r is None for r in l2_results.values()):
        return None

    # ── Level 3 (root): aggregate L2 outputs ──
    l2_output_keys = [
        l2_results[i]['response']['output_key']
        for i in range(num_l2)
    ]

    root_payload = {
        'mode': 'lifl-level',
        'bucket': bucket,
        'model_name': model_name,
        'level': 3,
        'agg_id': 0,
        'input_keys': l2_output_keys,
        'num_inputs_total': len(l2_output_keys),
        'round_id': round_id,
    }
    root_result = invoke_lambda(lambda_client, func_name, root_payload)
    t_root_done = time.perf_counter()

    if root_result is None:
        return None

    wall_s = t_root_done - t0

    # Collect timings
    l1_timings = [l1_results[i]['response']['timings'] for i in range(num_l1)]
    l2_timings = [l2_results[i]['response']['timings'] for i in range(num_l2)]
    root_timings = root_result['response']['timings']

    # S3 ops: L1 reads N client grads + writes num_l1 results
    #         L2 reads num_l1 L1 results + writes num_l2 results
    #         Root reads num_l2 L2 results + writes 1 final
    total_s3_gets = NUM_CLIENTS + num_l1 + num_l2
    total_s3_puts = num_l1 + num_l2 + 1
    total_invocations = num_l1 + num_l2 + 1

    all_durations = (
        [t['total_s'] for t in l1_timings] +
        [t['total_s'] for t in l2_timings] +
        [root_timings['total_s']]
    )

    lambda_cost = sum(
        (memory_mb / 1024) * d * LAMBDA_PRICE_PER_GB_S
        for d in all_durations
    )
    s3_cost = (total_s3_gets / 1000) * S3_GET_PRICE_PER_1K + \
              (total_s3_puts / 1000) * S3_PUT_PRICE_PER_1K

    return {
        'architecture': 'lifl',
        'topology': {'branch': branch, 'num_l1': num_l1, 'num_l2': num_l2},
        'wall_clock_s': wall_s,
        'l1_phase_s': t_l1_done - t0,
        'l2_phase_s': t_l2_done - t_l1_done,
        'root_phase_s': t_root_done - t_l2_done,
        'l1_durations_s': [t['total_s'] for t in l1_timings],
        'l2_durations_s': [t['total_s'] for t in l2_timings],
        'root_duration_s': root_timings['total_s'],
        's3_bytes_read': (
            sum(t['s3_bytes_read'] for t in l1_timings) +
            sum(t['s3_bytes_read'] for t in l2_timings) +
            root_timings['s3_bytes_read']
        ),
        'num_lambda_invocations': total_invocations,
        'num_s3_gets': total_s3_gets,
        'num_s3_puts': total_s3_puts,
        'memory_mb': memory_mb,
        'cost': {
            'lambda_compute': lambda_cost,
            's3_io': s3_cost,
            'total': lambda_cost + s3_cost,
        },
    }


# ─── Main experiment runner ──────────────────────────────────────────────────

ARCHITECTURE_RUNNERS = {
    'grads_sharding': run_grads_sharding,
    'lambda_fl': run_lambda_fl,
    'lifl': run_lifl,
}


def _save_incremental(output_dir, model_name, model_results):
    """Save results incrementally so nothing is lost on crash."""
    model_file = output_dir / f"rq3_{model_name}.json"
    with open(model_file, 'w') as f:
        json.dump({model_name: model_results}, f, indent=2, default=str)
    print(f"  [saved → {model_file}]")


def run_model_experiment(lambda_client, registry, model_name, model_config, bucket,
                          num_rounds, num_reps, output_dir=None):
    """Run all feasible architectures for one model."""
    grad_mb = model_config['grad_mb']
    print(f"\n{'='*75}")
    print(f"  MODEL: {model_name} ({grad_mb:.1f} MB gradient)")
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {num_rounds} | Reps: {num_reps}")
    print(f"{'='*75}")

    model_results = {}

    for arch_name, runner in ARCHITECTURE_RUNNERS.items():
        arch_info = registry.get(model_name, {}).get(arch_name, {})

        if arch_info.get('infeasible'):
            streaming_mem = 2 * grad_mb
            print(f"\n  [{arch_name}] INFEASIBLE — needs {streaming_mem:.0f} MB streaming "
                  f"> {LAMBDA_MAX_MEMORY_MB} MB limit")
            model_results[arch_name] = {
                'architecture': arch_name,
                'feasible': False,
                'reason': arch_info.get('reason', f'Exceeds {LAMBDA_MAX_MEMORY_MB}MB'),
                'memory_needed_mb': streaming_mem,
            }
            if output_dir:
                _save_incremental(output_dir, model_name, model_results)
            continue

        if not arch_info.get('function_name'):
            print(f"\n  [{arch_name}] SKIPPED — no function registered")
            continue

        print(f"\n  [{arch_name}] Running (memory={arch_info['memory_mb']}MB)...")

        arch_runs = []
        for round_id in range(num_rounds):
            for rep in range(num_reps):
                label = f"r{round_id}/rep{rep}"
                print(f"    {label}...", end=" ", flush=True)

                try:
                    result = runner(lambda_client, registry, model_name, model_config,
                                    bucket, round_id)
                except Exception as e:
                    print(f"EXCEPTION: {e}")
                    result = None

                if result is None:
                    print("FAILED")
                    continue

                is_cold = (round_id == 0 and rep == 0)
                result['round_id'] = round_id
                result['rep'] = rep
                result['cold_start'] = is_cold

                wall = result['wall_clock_s']
                cost = result['cost']['total']
                print(f"wall={wall:.2f}s | cost=${cost:.6f}"
                      + (" [cold]" if is_cold else ""))

                arch_runs.append(result)

        if not arch_runs:
            model_results[arch_name] = {
                'architecture': arch_name,
                'feasible': True,
                'error': 'All invocations failed',
            }
            if output_dir:
                _save_incremental(output_dir, model_name, model_results)
            continue

        # Summary (warm only)
        warm = [r for r in arch_runs if not r.get('cold_start', False)]
        if not warm:
            warm = arch_runs

        wall_times = [r['wall_clock_s'] for r in warm]
        costs = [r['cost']['total'] for r in warm]
        lambda_costs = [r['cost']['lambda_compute'] for r in warm]
        s3_costs = [r['cost']['s3_io'] for r in warm]

        summary = {
            'architecture': arch_name,
            'feasible': True,
            'memory_mb': arch_info['memory_mb'],
            'num_warm_runs': len(warm),
            'wall_clock_s': {'mean': float(np.mean(wall_times)), 'std': float(np.std(wall_times)),
                             'min': float(np.min(wall_times)), 'max': float(np.max(wall_times))},
            'cost_per_round': {
                'total': {'mean': float(np.mean(costs)), 'std': float(np.std(costs))},
                'lambda_compute': {'mean': float(np.mean(lambda_costs))},
                's3_io': {'mean': float(np.mean(s3_costs))},
            },
            'cost_per_1000_rounds': float(np.mean(costs)) * 1000,
            'num_lambda_invocations': warm[0].get('num_lambda_invocations', 0),
            'num_s3_gets': warm[0].get('num_s3_gets', 0),
            'num_s3_puts': warm[0].get('num_s3_puts', 0),
            'all_runs': arch_runs,
        }

        # Cold start
        cold = [r for r in arch_runs if r.get('cold_start')]
        if cold:
            summary['cold_start_wall_s'] = cold[0]['wall_clock_s']

        print(f"\n    --- {arch_name} Summary (warm) ---")
        print(f"    Wall-clock: {np.mean(wall_times):.2f} ± {np.std(wall_times):.2f} s")
        print(f"    Cost/round: ${np.mean(costs):.6f}")
        print(f"    Cost/1K:    ${np.mean(costs)*1000:.4f}")

        model_results[arch_name] = summary

        # Save after each architecture completes
        if output_dir:
            _save_incremental(output_dir, model_name, model_results)

    return model_results


def print_comparison_table(all_results):
    """Print final cross-architecture comparison table."""
    print(f"\n\n{'#'*85}")
    print(f"  RQ3 FINAL COMPARISON TABLE")
    print(f"{'#'*85}")

    header = (f"{'Model':<16} {'Architecture':<18} {'Mem(MB)':>7} "
              f"{'Wall(s)':>8} {'Cost/rnd':>10} {'Cost/1K':>10} "
              f"{'Invocations':>11} {'S3 ops':>8}")
    print(f"  {header}")
    print(f"  {'-'*85}")

    for model_name, arch_results in all_results.items():
        for arch_name in ['grads_sharding', 'lambda_fl', 'lifl']:
            if arch_name not in arch_results:
                continue
            r = arch_results[arch_name]

            if not r.get('feasible', True):
                mem_needed = r.get('memory_needed_mb', '?')
                print(f"  {model_name:<16} {arch_name:<18} {'---':>7} "
                      f"{'INFEASIBLE':>8} {'---':>10} {'---':>10} "
                      f"{'---':>11} {'---':>8}  (needs {mem_needed:.0f}MB)")
                continue

            if 'error' in r:
                print(f"  {model_name:<16} {arch_name:<18} {'FAILED':>7}")
                continue

            wall = r['wall_clock_s']['mean']
            cost = r['cost_per_round']['total']['mean']
            cost_1k = r['cost_per_1000_rounds']
            mem = r['memory_mb']
            invocations = r['num_lambda_invocations']
            s3_ops = r['num_s3_gets'] + r['num_s3_puts']

            print(f"  {model_name:<16} {arch_name:<18} {mem:>7} "
                  f"{wall:>8.2f} ${cost:>9.6f} ${cost_1k:>9.4f} "
                  f"{invocations:>11} {s3_ops:>8}")

        print(f"  {'-'*85}")

    print(f"{'#'*85}")


def main():
    parser = argparse.ArgumentParser(description="Run RQ3 Lambda experiment")
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--model', default=None, choices=list(MODELS.keys()),
                        help='Single model (default: all)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 round, 1 rep')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/rq3_lambda/)')
    args = parser.parse_args()

    num_rounds = 1 if args.quick else NUM_ROUNDS
    num_reps = 1 if args.quick else NUM_REPS

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(os.path.dirname(__file__)) / '..' / 'results' / 'rq3_lambda'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load function registry
    registry = load_function_registry()

    # Lambda client with long timeout
    boto_config = Config(read_timeout=900, connect_timeout=10,
                         retries={'max_attempts': 3})
    lambda_client = boto3.client('lambda', region_name=args.region, config=boto_config)

    models = {args.model: MODELS[args.model]} if args.model else MODELS

    print(f"\n{'#'*75}")
    print(f"  RQ3: CROSS-ARCHITECTURE COMPARISON ON AWS LAMBDA")
    print(f"{'#'*75}")
    print(f"  Bucket:  {args.bucket}")
    print(f"  Region:  {args.region}")
    print(f"  Models:  {', '.join(models.keys())}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds:  {num_rounds} × {num_reps} reps")
    if args.quick:
        print(f"  ** QUICK MODE **")
    print(f"{'#'*75}")

    all_results = {}
    t_global = time.time()

    for model_name, model_config in models.items():
        model_results = run_model_experiment(
            lambda_client, registry, model_name, model_config,
            args.bucket, num_rounds, num_reps, output_dir=output_dir
        )
        all_results[model_name] = model_results

    elapsed = time.time() - t_global

    # Print comparison table
    print_comparison_table(all_results)

    # Save combined results
    combined = {
        'experiment': 'RQ3: Cross-Architecture Comparison on AWS Lambda',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'bucket': args.bucket,
            'region': args.region,
            'num_clients': NUM_CLIENTS,
            'num_rounds': num_rounds,
            'num_reps': num_reps,
            'lambda_max_memory_mb': LAMBDA_MAX_MEMORY_MB,
            'quick_mode': args.quick,
        },
        'models': {name: {
            'params': cfg['params'],
            'grad_mb': cfg['grad_mb'],
            'num_shards': cfg['num_shards'],
        } for name, cfg in models.items()},
        'results': all_results,
        'total_experiment_time_s': elapsed,
    }

    combined_file = output_dir / 'rq3_lambda_results.json'
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nTotal experiment time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Combined results saved: {combined_file}")


if __name__ == '__main__':
    main()
