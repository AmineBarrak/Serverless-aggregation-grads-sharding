"""
Core runner: executes any FL system in a subprocess with proper path setup.

Avoids module import issues by running each system with explicit sys.path injection.
All RQ scripts should use `run_system()` from this module.

Windows-compatible: uses forward slashes in all embedded paths.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Convert to forward slashes for safe embedding in Python strings on all OSes
# (Windows Python accepts forward slashes just fine)
_ROOT = str(PROJECT_ROOT).replace("\\", "/")


def _build_code(system: str, num_clients: int, num_rounds: int,
                model_name: str, dataset_name: str, num_shards: int,
                local_epochs: int, batch_size: int = 32) -> str:
    """Build the Python code string to execute in a subprocess."""

    # Common preamble: path setup (uses forward slashes, safe on Windows)
    preamble = f'''import sys, time, json, os, logging, torch, gc
logging.disable(logging.CRITICAL)
ROOT = "{_ROOT}"
sys.path.insert(0, ROOT)

# Clear GPU cache before each run to avoid OOM from prior allocations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
'''

    config_block = f'''
config = FLConfig(
    num_clients={num_clients}, num_rounds={num_rounds},
    local_epochs={local_epochs}, batch_size={batch_size}, learning_rate=0.01,
    model_name="{model_name}", dataset_name="{dataset_name}",
    num_shards={num_shards}, seed=42,
)
'''

    if system == "lambda-fl":
        return preamble + f'''sys.path.insert(0, os.path.join(ROOT, "lambda-fl"))
from shared.config import FLConfig
from server import LambdaFLServer
{config_block}
t0 = time.time()
server = LambdaFLServer(config)
model, metrics = server.run()
elapsed = time.time() - t0

agg_times = [r.get("aggregation_metrics", {{}}).get("total_time", 0) for r in metrics.get("rounds", [])]
round_times = [r.get("round_time", 0) for r in metrics.get("rounds", [])]
avg_agg = sum(agg_times) / len(agg_times) if agg_times else 0
avg_round = sum(round_times) / len(round_times) if round_times else 0

result = {{
    "system": "lambda-fl",
    "num_clients": {num_clients},
    "num_rounds": {num_rounds},
    "model_name": "{model_name}",
    "elapsed_time_s": elapsed,
    "avg_round_latency_s": avg_round,
    "avg_aggregation_latency_s": avg_agg,
    "total_lambda_seconds": sum(agg_times),
    "total_cost_usd": sum(agg_times) * 0.5 * 0.0000166667,
    "per_round": metrics.get("rounds", []),
}}
print("__RESULT__" + json.dumps(result))
'''

    elif system == "lifl":
        return preamble + f'''sys.path.insert(0, os.path.join(ROOT, "lifl"))
from shared.config import FLConfig
from server import LIFLServer
{config_block}
t0 = time.time()
server = LIFLServer(config)
model, metrics = server.run()
elapsed = time.time() - t0

# Extract average aggregation latency from per-round metrics
agg_latencies = [r["aggregation_latency_s"] for r in metrics.per_round_metrics if "aggregation_latency_s" in r]
avg_agg_latency = sum(agg_latencies) / len(agg_latencies) if agg_latencies else 0

result = {{
    "system": "lifl",
    "num_clients": {num_clients},
    "num_rounds": {num_rounds},
    "model_name": "{model_name}",
    "elapsed_time_s": elapsed,
    "avg_round_latency_s": elapsed / {num_rounds},
    "avg_aggregation_latency_s": avg_agg_latency,
    "total_lambda_seconds": elapsed * 0.3,
    "total_cost_usd": elapsed * 0.3 * 0.5 * 0.0000166667,
    "metrics": metrics if isinstance(metrics, dict) else {{}},
}}
print("__RESULT__" + json.dumps(result))
'''

    elif system == "grads-sharding":
        return preamble + f'''sys.path.insert(0, os.path.join(ROOT, "grads-sharding"))
from shared.config import FLConfig
from server import GradShardingServer
{config_block}
t0 = time.time()
server = GradShardingServer(config)
model, summary = server.run()
elapsed = time.time() - t0

per_round = summary.get("metrics", {{}}).get("per_round_metrics", [])
avg_round = sum(r.get("latency_s", 0) for r in per_round) / len(per_round) if per_round else elapsed / {num_rounds}
avg_agg = sum(r.get("aggregation_latency_s", 0) for r in per_round) / len(per_round) if per_round else 0
avg_mem = sum(r.get("peak_memory_mb", 0) for r in per_round) / len(per_round) if per_round else 0

result = {{
    "system": "grads-sharding",
    "num_clients": {num_clients},
    "num_rounds": {num_rounds},
    "num_shards": {num_shards},
    "model_name": "{model_name}",
    "elapsed_time_s": elapsed,
    "avg_round_latency_s": avg_round,
    "avg_aggregation_latency_s": avg_agg,
    "avg_peak_memory_mb": avg_mem,
    "total_cost_usd": summary.get("total_cost_usd", 0),
    "cost_breakdown": summary.get("cost_breakdown", {{}}),
    "per_round": per_round,
}}
print("__RESULT__" + json.dumps(result))
'''

    else:
        raise ValueError(f"Unknown system: {system}")


def run_system(
    system: str,
    num_clients: int,
    num_rounds: int,
    model_name: str = "simple_cnn",
    dataset_name: str = "cifar100",
    num_shards: int = 4,
    local_epochs: int = 1,
    batch_size: int = 32,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single FL system and collect metrics.

    Executes each system in a clean subprocess to avoid import collisions.
    Works on both Windows and Linux.

    Args:
        system: One of 'lambda-fl', 'lifl', 'grads-sharding'
        num_clients: Number of FL clients
        num_rounds: Number of training rounds
        model_name: Model to use (simple_cnn, resnet18, etc.)
        dataset_name: Dataset (cifar100, femnist, rvlcdip)
        num_shards: Shards for grads-sharding (ignored by others)
        local_epochs: Local training epochs per round
        batch_size: Batch size for local training (reduce for large models)
        output_file: Optional path to save metrics JSON

    Returns:
        Dictionary with timing and metrics, or error info on failure.
    """
    try:
        code = _build_code(system, num_clients, num_rounds, model_name,
                           dataset_name, num_shards, local_epochs, batch_size)
    except ValueError as e:
        return {"system": system, "error": str(e)}

    # Execute in subprocess
    try:
        print(f"  Running {system} ({num_clients} clients, {num_rounds} rounds)...")
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour — large models (VGG-16) with many clients need more time
            cwd=str(PROJECT_ROOT),
        )
        wall_time = time.time() - start

        if result.returncode != 0:
            err_msg = result.stderr[-500:] if result.stderr else "No stderr"
            print(f"    FAILED ({wall_time:.1f}s): {err_msg[:150]}")
            return {
                "system": system,
                "num_clients": num_clients,
                "error": err_msg,
                "elapsed_time_s": wall_time,
            }

        # Parse the __RESULT__ line from stdout
        for line in reversed(result.stdout.strip().split("\n")):
            if line.startswith("__RESULT__"):
                data = json.loads(line[len("__RESULT__"):])
                print(f"    OK ({data.get('elapsed_time_s', wall_time):.1f}s)")

                # Save to file if requested
                if output_file:
                    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, "w") as f:
                        json.dump(data, f, indent=2)

                return data

        print(f"    WARNING: No result marker found in output")
        return {
            "system": system,
            "num_clients": num_clients,
            "elapsed_time_s": wall_time,
            "warning": "No __RESULT__ marker",
        }

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (3600s)")
        return {"system": system, "num_clients": num_clients, "error": "Timeout"}
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        return {"system": system, "num_clients": num_clients, "error": str(e)}
