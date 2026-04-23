"""
RQ1: Serverless Motivation — Parameter Server Idle Time Analysis
=================================================================

Motivation:
  In federated learning (FedAvg), each round consists of two phases:
    (1) Client training: N clients train locally for E epochs on their private data
    (2) Server aggregation: the PS collects N gradient updates and averages them

  The server is IDLE during the entire client training phase — it has nothing to
  do until all clients finish. If the server is idle 90%+ of each round, paying
  for a persistent server is wasteful. This motivates serverless aggregation,
  where compute is provisioned only for the brief aggregation window.

Experimental Setup:
  We simulate a realistic cross-silo FL deployment with the following parameters:

  - N = 20 clients per round (typical cross-silo setting)
  - Dataset: CIFAR-10 (50,000 training samples), IID partitioned across N clients
    → Each client holds |D_k| = 2,500 samples
  - Local epochs: E = 5 (standard FedAvg configuration per McMahan et al. 2017)
  - Batch size: B = 32
  - Local steps per client per round: ⌈|D_k| / B⌉ × E = ⌈2500/32⌉ × 5 = 395 steps
  - For language models: equivalent token budget (~400K tokens per client, B=8, seq_len=128)

  Models tested (same as paper evaluation):
    - ResNet-18    (11.7M params,   45 MB gradient)  — lightweight baseline
    - VGG-16       (138M params,   528 MB gradient)  — medium vision model
    - GPT-2 Medium (345M params, 1,320 MB gradient)  — transformer LM
    - GPT-2 Large  (774M params, 2,960 MB gradient)  — large transformer LM

  Client training runs on GPU (simulating a cross-silo client with GPU resources).
  Server aggregation runs on CPU (standard PS deployment — receives, sums, divides).

  We measure:
    - T_train: wall-clock time for one client to complete E local epochs
    - T_agg:   wall-clock time for the server to aggregate N client gradients
    - PS idle ratio = T_train / (T_train + T_agg)
      (In parallel FL, the server waits for the slowest client ≈ T_train)

Usage:
  # Run all models sequentially
  python rq1_ps_idle_time.py

  # Run single model (for SLURM job array)
  python rq1_ps_idle_time.py --model_idx 0   # ResNet-18
  python rq1_ps_idle_time.py --model_idx 1   # VGG-16
  python rq1_ps_idle_time.py --model_idx 2   # GPT-2 Medium
  python rq1_ps_idle_time.py --model_idx 3   # GPT-2 Large

  # Merge parallel results
  python rq1_ps_idle_time.py --merge
"""

import argparse
import json
import os
import time
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


# ─── FL Configuration ─────────────────────────────────────────────────────────
# These match standard FedAvg settings (McMahan et al., 2017)

NUM_CLIENTS = 20               # N: clients per round (cross-silo)
CIFAR10_TRAIN_SIZE = 50000     # Total CIFAR-10 training samples
SAMPLES_PER_CLIENT = CIFAR10_TRAIN_SIZE // NUM_CLIENTS  # |D_k| = 2,500
LOCAL_EPOCHS = 5               # E: local epochs per round
BATCH_SIZE_VISION = 32         # B: batch size for vision models
BATCH_SIZE_LM = 8              # B: batch size for language models
SEQ_LEN = 128                  # Sequence length for language models

# Derived: local steps per round
STEPS_VISION = math.ceil(SAMPLES_PER_CLIENT / BATCH_SIZE_VISION) * LOCAL_EPOCHS  # 395
STEPS_LM = math.ceil(SAMPLES_PER_CLIENT / BATCH_SIZE_LM) * LOCAL_EPOCHS         # 1565
# For LMs we use a token-equivalent budget: ~400K tokens ≈ 390 steps at B=8, seq=128
STEPS_LM_EQUIV = math.ceil(400_000 / (BATCH_SIZE_LM * SEQ_LEN)) * 1  # ~390 steps (1 pass)
# Use the same 395 steps as vision for fair comparison across architectures
STEPS_PER_ROUND = STEPS_VISION  # 395 steps for all models


# ─── Model Definitions ───────────────────────────────────────────────────────

def build_resnet18():
    """ResNet-18: ~11.7M parameters, ~45 MB gradient."""
    from torchvision.models import resnet18
    model = resnet18(num_classes=10)
    dummy_input = torch.randn(BATCH_SIZE_VISION, 3, 32, 32)  # CIFAR-10: 32×32
    dummy_target = torch.randint(0, 10, (BATCH_SIZE_VISION,))
    return model, dummy_input, dummy_target, "ResNet-18", STEPS_PER_ROUND


def build_vgg16():
    """VGG-16: ~138M parameters, ~528 MB gradient."""
    from torchvision.models import vgg16
    model = vgg16(num_classes=10)
    dummy_input = torch.randn(BATCH_SIZE_VISION, 3, 224, 224)  # ImageNet-size input
    dummy_target = torch.randint(0, 10, (BATCH_SIZE_VISION,))
    return model, dummy_input, dummy_target, "VGG-16", STEPS_PER_ROUND


def build_gpt2_medium():
    """GPT-2 Medium: ~345M parameters, ~1,320 MB gradient."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config(
            vocab_size=50257, n_positions=1024,
            n_embd=1024, n_layer=24, n_head=16,
        )
        model = GPT2LMHeadModel(config)
        dummy_input = torch.randint(0, 50257, (BATCH_SIZE_LM, SEQ_LEN))
        dummy_target = torch.randint(0, 50257, (BATCH_SIZE_LM, SEQ_LEN))
        return model, dummy_input, dummy_target, "GPT-2 Medium", STEPS_PER_ROUND
    except ImportError:
        print("ERROR: transformers package required for GPT-2 models.")
        print("Install with: pip install transformers")
        sys.exit(1)


def build_gpt2_large():
    """GPT-2 Large: ~774M parameters, ~2,960 MB gradient."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config(
            vocab_size=50257, n_positions=1024,
            n_embd=1280, n_layer=36, n_head=20,
        )
        model = GPT2LMHeadModel(config)
        dummy_input = torch.randint(0, 50257, (BATCH_SIZE_LM, SEQ_LEN))
        dummy_target = torch.randint(0, 50257, (BATCH_SIZE_LM, SEQ_LEN))
        return model, dummy_input, dummy_target, "GPT-2 Large", STEPS_PER_ROUND
    except ImportError:
        print("ERROR: transformers package required for GPT-2 models.")
        print("Install with: pip install transformers")
        sys.exit(1)


# ─── Measurement Functions ────────────────────────────────────────────────────

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradient_size_mb(model):
    """Gradient size in MB (float32)."""
    return count_parameters(model) * 4 / (1024 * 1024)


def measure_client_training(model, dummy_input, dummy_target, device,
                            local_steps, warmup_steps=3):
    """
    Measure wall-clock time for ONE client to complete E local epochs.
    This simulates: client receives global model, trains for local_steps
    iterations on local data, produces gradient update.
    """
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    dummy_target = dummy_target.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    def train_step():
        optimizer.zero_grad()
        output = model(dummy_input)
        # HuggingFace models return objects; extract logits
        if hasattr(output, 'logits'):
            output = output.logits
        elif isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3:  # Language model: (B, seq, vocab)
            output = output.view(-1, output.size(-1))
            target = dummy_target.view(-1)
        else:
            target = dummy_target
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Warmup GPU (JIT compilation, CUDA context, etc.)
    for _ in range(warmup_steps):
        train_step()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run: full local training (E epochs = local_steps iterations)
    reps = 3  # Repeat for stability
    times = []
    for _ in range(reps):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(local_steps):
            train_step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    model = model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "total_mean_s": float(np.mean(times)),
        "total_std_s": float(np.std(times)),
        "per_step_ms": float(np.mean(times) / local_steps * 1000),
        "local_steps": local_steps,
        "all_times_s": times,
    }


def measure_aggregation(model, num_clients, device_agg='cpu'):
    """
    Measure server-side FedAvg aggregation time.

    The server performs: θ_global = (1/N) Σ θ_k
    This is element-wise addition of N gradient vectors followed by scalar division.

    We pre-allocate all client gradients before timing to isolate the pure
    compute cost of aggregation (in real FL, gradients arrive over the network;
    we measure only the aggregation compute, not network transfer).
    """
    total_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    grad_mb = total_params * 4 / (1024 * 1024)

    print(f"    Pre-allocating {num_clients} gradient vectors ({grad_mb:.1f} MB each)...")

    # Pre-allocate: simulates gradients already received from all clients
    client_grads = [torch.randn(total_params, dtype=torch.float32, device=device_agg)
                    for _ in range(num_clients)]

    warmup = 2
    reps = 5
    times = []

    for rep in range(warmup + reps):
        agg_buffer = torch.zeros(total_params, dtype=torch.float32, device=device_agg)

        # Time ONLY the aggregation: sum N vectors + divide by N
        t0 = time.perf_counter()
        for c in range(num_clients):
            agg_buffer.add_(client_grads[c])
        agg_buffer.div_(num_clients)
        t1 = time.perf_counter()

        if rep >= warmup:
            times.append(t1 - t0)
        del agg_buffer

    del client_grads
    if device_agg == 'cuda':
        torch.cuda.empty_cache()

    return {
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "mean_ms": float(np.mean(times) * 1000),
        "all_times_s": times,
    }


# ─── Experiment Runners ───────────────────────────────────────────────────────

MODEL_BUILDERS = [
    build_resnet18,
    build_vgg16,
    build_gpt2_medium,
    build_gpt2_large,
]


def run_single_model(args, model_idx):
    """Run experiment for a single model (for SLURM job array)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if model_idx < 0 or model_idx >= len(MODEL_BUILDERS):
        print(f"ERROR: model_idx {model_idx} out of range [0, {len(MODEL_BUILDERS)-1}]")
        sys.exit(1)

    builder = MODEL_BUILDERS[model_idx]
    model, dummy_input, dummy_target, model_name, local_steps = builder()
    n_params = count_parameters(model)
    grad_mb = gradient_size_mb(model)

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"  Parameters:    {n_params:,} ({grad_mb:.1f} MB gradient)")
    print(f"  Num clients:   N = {args.num_clients}")
    print(f"  Local epochs:  E = {LOCAL_EPOCHS}")
    print(f"  Samples/client:|D_k| = {SAMPLES_PER_CLIENT}")
    print(f"  Local steps:   ceil({SAMPLES_PER_CLIENT}/{BATCH_SIZE_VISION}) x {LOCAL_EPOCHS} = {local_steps}")
    print(f"{'='*70}")

    # 1. Client training
    print(f"\n  [Phase 1] Measuring client local training ({local_steps} steps)...")
    try:
        client_timing = measure_client_training(
            model, dummy_input, dummy_target, device,
            local_steps=local_steps, warmup_steps=3
        )
        print(f"    → Client training time: {client_timing['total_mean_s']*1000:.1f} ms "
              f"({client_timing['per_step_ms']:.2f} ms/step)")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    ERROR: GPU OOM for {model_name}")
            torch.cuda.empty_cache()
            sys.exit(1)
        raise

    # 2. Server aggregation
    print(f"\n  [Phase 2] Measuring server FedAvg aggregation (N={args.num_clients} clients, CPU)...")
    agg_timing = measure_aggregation(model, args.num_clients, device_agg='cpu')
    print(f"    → Aggregation time: {agg_timing['mean_ms']:.1f} ms")

    # 3. Compute PS idle ratio
    t_train = client_timing['total_mean_s']
    t_agg = agg_timing['mean_s']
    t_round = t_train + t_agg
    ps_idle_pct = (t_train / t_round) * 100

    print(f"\n  [Result] Round breakdown:")
    print(f"    T_train (client):      {t_train*1000:>10.1f} ms ({ps_idle_pct:.1f}% of round)")
    print(f"    T_agg   (server):      {t_agg*1000:>10.1f} ms ({100-ps_idle_pct:.1f}% of round)")
    print(f"    T_round (total):       {t_round*1000:>10.1f} ms")
    print(f"    ═══ PS idle ratio:     {ps_idle_pct:.1f}% ═══")

    result = {
        "model_name": model_name,
        "num_parameters": n_params,
        "gradient_size_mb": round(grad_mb, 1),
        "config": {
            "num_clients": args.num_clients,
            "local_epochs": LOCAL_EPOCHS,
            "samples_per_client": SAMPLES_PER_CLIENT,
            "batch_size": BATCH_SIZE_VISION if "GPT" not in model_name else BATCH_SIZE_LM,
            "local_steps": local_steps,
        },
        "client_training": {
            "total_mean_ms": round(t_train * 1000, 2),
            "total_std_ms": round(client_timing['total_std_s'] * 1000, 2),
            "per_step_ms": round(client_timing['per_step_ms'], 2),
        },
        "aggregation": {
            "mean_ms": round(agg_timing['mean_ms'], 2),
            "std_ms": round(agg_timing['std_s'] * 1000, 2),
        },
        "round_timing": {
            "t_train_ms": round(t_train * 1000, 2),
            "t_agg_ms": round(t_agg * 1000, 2),
            "t_round_ms": round(t_round * 1000, 2),
            "ps_idle_pct": round(ps_idle_pct, 1),
        },
    }

    # Save per-model result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_file = output_dir / f"rq1_model_{model_idx}_{safe_name}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {output_file}")


def run_all(args):
    """Run all models sequentially."""
    for idx in range(len(MODEL_BUILDERS)):
        run_single_model(args, idx)
    merge_results(args)


def merge_results(args):
    """Merge per-model JSON files into final combined result."""
    output_dir = Path(args.output_dir)
    model_files = sorted(output_dir.glob("rq1_model_*.json"))

    if not model_files:
        print(f"No per-model result files found in {output_dir}")
        sys.exit(1)

    results = []
    for f in model_files:
        with open(f) as fh:
            results.append(json.load(fh))

    combined = {
        "experiment": "RQ1: Parameter Server Idle Time",
        "description": (
            "Measures the fraction of each FL round where the parameter server "
            "is idle (waiting for clients to finish local training). "
            "High idle ratios motivate serverless aggregation."
        ),
        "fl_config": {
            "num_clients_N": NUM_CLIENTS,
            "local_epochs_E": LOCAL_EPOCHS,
            "dataset": "CIFAR-10 (IID partition)",
            "samples_per_client": SAMPLES_PER_CLIENT,
            "batch_size_vision": BATCH_SIZE_VISION,
            "batch_size_lm": BATCH_SIZE_LM,
            "local_steps": STEPS_PER_ROUND,
        },
        "results": results,
    }

    output_file = output_dir / "rq1_ps_idle_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Merged {len(results)} models → {output_file}")
    print(f"\nRQ1 SUMMARY: PS Idle Time per FL Round")
    print(f"  N={NUM_CLIENTS} clients, E={LOCAL_EPOCHS} local epochs, "
          f"|D_k|={SAMPLES_PER_CLIENT} samples/client, {STEPS_PER_ROUND} steps/round")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Params':>10} {'Grad':>8} {'T_train':>10} {'T_agg':>8} {'PS Idle':>8}")
    print(f"{'-'*62}")
    for r in results:
        n = r['num_parameters']
        p_str = f"{n/1e6:.0f}M" if n < 1e9 else f"{n/1e9:.1f}B"
        print(f"{r['model_name']:<18} {p_str:>10} {r['gradient_size_mb']:>6.0f}MB "
              f"{r['round_timing']['t_train_ms']:>8.0f}ms {r['round_timing']['t_agg_ms']:>6.0f}ms "
              f"{r['round_timing']['ps_idle_pct']:>7.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1: PS Idle Time — Motivation for Serverless FL Aggregation")
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS,
                        help=f"Clients per round (default: {NUM_CLIENTS})")
    parser.add_argument("--output_dir", type=str, default="results/rq1_ps_idle_time",
                        help="Output directory")
    parser.add_argument("--model_idx", type=int, default=-1,
                        help="Single model index (0-3). -1 = all sequential")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-model results into final JSON")
    args = parser.parse_args()

    if args.merge:
        merge_results(args)
    elif args.model_idx >= 0:
        run_single_model(args, args.model_idx)
    else:
        run_all(args)
