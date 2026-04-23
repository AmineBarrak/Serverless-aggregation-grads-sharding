"""
Generate Lambda deployment figures for the paper.

Figures:
  fig_lambda_breakdown:    Time breakdown (S3 read / compute / S3 write) for 4 models
  fig_vgg16_shard_sweep:   (a) Time breakdown by M, (b) Cost breakdown by M
  fig_vgg16_shard_tradeoff: (a) Latency vs cost, (b) S3 ops & throughput vs M

Data source: experiments/lambda_validation/results/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "IC2E2026-ServerlessScalability" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(__file__).parent / "lambda_validation" / "results"

# IEEE column width styling
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

# ── AWS Pricing ──
S3_PUT_PER_OP = 0.000005       # $0.005 per 1K PUTs
S3_GET_PER_OP = 0.0000004      # $0.0004 per 1K GETs
LAMBDA_PER_GB_S = 0.0000166667


def s3_cost_per_round(N, M):
    """Full-round S3 cost: (NM+M) PUTs + 2NM GETs."""
    puts = N * M + M
    gets = 2 * N * M
    return puts * S3_PUT_PER_OP + gets * S3_GET_PER_OP


def s3_ops_per_round(N, M):
    """Full-round S3 ops: 3NM + M."""
    return 3 * N * M + M


# ── Load data ──

def load_validation_results():
    """Load the 4-model Lambda validation results."""
    f = RESULTS_DIR / "lambda_validation_all.json"
    with open(f) as fh:
        return json.load(fh)["results"]


def load_shard_sweep():
    """Load VGG-16 shard sweep results."""
    f = RESULTS_DIR / "vgg16_shard_sweep" / "vgg16_shard_sweep_results.json"
    with open(f) as fh:
        return json.load(fh)


# ── Figure 1: Lambda Breakdown (4 models) ──

def generate_lambda_breakdown():
    """Stacked bar: S3 read / compute / S3 write for each model."""
    data = load_validation_results()

    models = ["resnet18_m1", "vgg16_m1", "gpt2_medium_m4", "gpt2_large_m4"]
    labels = ["ResNet-18\n($M$=1)", "VGG-16\n($M$=1)", "GPT-2 Med\n($M$=4)", "GPT-2 Large\n($M$=4)"]

    s3_read = [data[m]["s3_read_ms"]["mean"] / 1000 for m in models]
    compute = [data[m]["compute_ms"]["mean"] / 1000 for m in models]
    s3_write = [data[m]["s3_write_ms"]["mean"] / 1000 for m in models]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(models))
    width = 0.5

    ax.bar(x, s3_read, width, label='S3 Read', color='#4A90D9', zorder=3)
    ax.bar(x, compute, width, bottom=s3_read, label='FedAvg Compute', color='#F5A623', zorder=3)
    ax.bar(x, s3_write, width, bottom=[r + c for r, c in zip(s3_read, compute)],
           label='S3 Write', color='#7B68EE', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Aggregation time (s)')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_lambda_breakdown.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig_lambda_breakdown.png", dpi=300, bbox_inches='tight')
    print(f"Saved fig_lambda_breakdown to {OUTPUT_DIR}")
    plt.close(fig)


# ── Figure 2: VGG-16 Shard Sweep ──

def generate_vgg16_shard_sweep():
    """(a) Time breakdown by M with speedup annotations, (b) Cost breakdown by M."""
    sweep = load_shard_sweep()
    results = sweep["results"]
    N = sweep["config"]["num_clients"]

    shard_counts = [r["num_shards"] for r in results]
    s3_read = [r["s3_read_ms"]["mean"] / 1000 for r in results]
    compute = [r["compute_ms"]["mean"] / 1000 for r in results]
    s3_write = [r["s3_write_ms"]["mean"] / 1000 for r in results]

    # Speedup relative to M=1 (use aggregation_ms for consistency with paper)
    agg_times = [r["aggregation_ms"]["mean"] / 1000 for r in results]
    speedups = [agg_times[0] / t for t in agg_times]

    # Costs per 1K rounds
    lambda_costs = []
    s3_costs = []
    for r in results:
        M = r["num_shards"]
        memory_gb = r["memory_mb"] / 1024
        mean_agg_s = r["aggregation_ms"]["mean"] / 1000
        lc = memory_gb * mean_agg_s * LAMBDA_PER_GB_S * M
        sc = s3_cost_per_round(N, M)
        lambda_costs.append(lc * 1000)
        s3_costs.append(sc * 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) Time breakdown
    x = np.arange(len(shard_counts))
    width = 0.5
    ax1.bar(x, s3_read, width, label='S3 Read', color='#4A90D9', zorder=3)
    ax1.bar(x, compute, width, bottom=s3_read, label='Compute', color='#F5A623', zorder=3)
    ax1.bar(x, s3_write, width, bottom=[r + c for r, c in zip(s3_read, compute)],
            label='S3 Write', color='#7B68EE', zorder=3)

    for i, sp in enumerate(speedups):
        total = s3_read[i] + compute[i] + s3_write[i]
        ax1.text(i, total + 3, f"{sp:.1f}$\\times$", ha='center', va='bottom', fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m) for m in shard_counts])
    ax1.set_xlabel('Shard count $M$')
    ax1.set_ylabel('Aggregation time (s)')
    ax1.set_title('(a) Time breakdown', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=6, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Cost breakdown
    ax2.bar(x, lambda_costs, width, label='Lambda compute', color='#1f77b4', zorder=3)
    ax2.bar(x, s3_costs, width, bottom=lambda_costs, label='S3 I/O', color='#ff7f0e', zorder=3)

    for i in range(len(shard_counts)):
        total = lambda_costs[i] + s3_costs[i]
        ax2.text(i, total + 0.2, f"${total:.2f}", ha='center', va='bottom', fontsize=6.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(m) for m in shard_counts])
    ax2.set_xlabel('Shard count $M$')
    ax2.set_ylabel('Cost per 1,000 rounds (USD)')
    ax2.set_title('(b) Cost breakdown', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=6, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_vgg16_shard_sweep.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig_vgg16_shard_sweep.png", dpi=300, bbox_inches='tight')
    print(f"Saved fig_vgg16_shard_sweep to {OUTPUT_DIR}")
    plt.close(fig)


# ── Figure 3: VGG-16 Shard Tradeoff ──

def generate_vgg16_shard_tradeoff():
    """(a) Latency vs cost scatter, (b) S3 ops & throughput vs M."""
    sweep = load_shard_sweep()
    results = sweep["results"]
    N = sweep["config"]["num_clients"]

    shard_counts = [r["num_shards"] for r in results]
    latencies = [r["aggregation_ms"]["mean"] / 1000 for r in results]
    throughputs = [r["s3_throughput_mbps"] for r in results]
    ops = [s3_ops_per_round(N, M) for M in shard_counts]

    # Costs per 1K rounds
    total_costs = []
    for r in results:
        M = r["num_shards"]
        memory_gb = r["memory_mb"] / 1024
        mean_agg_s = r["aggregation_ms"]["mean"] / 1000
        lc = memory_gb * mean_agg_s * LAMBDA_PER_GB_S * M * 1000
        sc = s3_cost_per_round(N, M) * 1000
        total_costs.append(lc + sc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) Latency vs Cost
    ax1.scatter(latencies, total_costs, c='#d62728', s=60, zorder=3, edgecolors='black', linewidth=0.5)
    for i, M in enumerate(shard_counts):
        ax1.annotate(f"$M$={M}", (latencies[i], total_costs[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=7)

    ax1.set_xlabel('Aggregation latency (s)')
    ax1.set_ylabel('Cost per 1,000 rounds (USD)')
    ax1.set_title('(a) Latency vs. cost', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # (b) S3 ops and throughput vs M
    color_ops = '#1f77b4'
    color_tp = '#d62728'

    ax2.bar(np.arange(len(shard_counts)), ops, 0.5, color=color_ops, alpha=0.7,
            label='S3 ops/round', zorder=3)
    ax2.set_xlabel('Shard count $M$')
    ax2.set_ylabel('S3 operations per round', color=color_ops)
    ax2.set_xticks(np.arange(len(shard_counts)))
    ax2.set_xticklabels([str(m) for m in shard_counts])
    ax2.tick_params(axis='y', labelcolor=color_ops)
    ax2.set_title('(b) S3 ops & throughput', fontsize=9, fontweight='bold')

    ax2b = ax2.twinx()
    ax2b.plot(np.arange(len(shard_counts)), throughputs, 'D-', color=color_tp,
              markersize=5, linewidth=1.5, label='Throughput', zorder=4)
    ax2b.set_ylabel('S3 throughput (MB/s)', color=color_tp)
    ax2b.tick_params(axis='y', labelcolor=color_tp)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper left')

    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_vgg16_shard_tradeoff.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig_vgg16_shard_tradeoff.png", dpi=300, bbox_inches='tight')
    print(f"Saved fig_vgg16_shard_tradeoff to {OUTPUT_DIR}")
    plt.close(fig)


if __name__ == "__main__":
    generate_lambda_breakdown()
    generate_vgg16_shard_sweep()
    generate_vgg16_shard_tradeoff()
    print("Done!")
