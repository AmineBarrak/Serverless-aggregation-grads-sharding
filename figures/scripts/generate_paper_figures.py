"""
Generate the two paper figures for RQ2 Part B with updated cost numbers.

fig_cost_vs_model: Total cost over 1,000 rounds vs model size (3 architectures)
fig_cost_breakdown: Lambda vs S3 cost breakdown for GradsSharding across shard counts
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "IC2E2026-ServerlessScalability" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ── Updated cost data from recalculation ──

# Model sizes in MB
MODEL_SIZES = [45, 528, 2048, 10240]
MODEL_LABELS = ["ResNet-18\n(45 MB)", "VGG-16\n(528 MB)", "GPT-2 Large\n(2,048 MB)", "10B-scale\n(10,240 MB)"]

# Costs per 1,000 rounds (N=50, GradsSharding M=4)
LAMBDA_FL_COST = [0.32, 1.45, 17.51, None]   # None = infeasible
LIFL_COST      = [0.36, 1.47, 17.27, None]
GS_COST        = [1.11, 1.33, 4.52,  86.44]

# ── RQ2 cost breakdown data (GradsSharding only) ──
# Full-round S3 ops per round: PUTs = NM + M, GETs = 2NM (agg reads + client read-back)
# Total ops = 3NM + M

SHARD_COUNTS = [1, 2, 4, 8, 16]
N_CLIENTS = 20

def _s3_cost_per_round(N, M):
    """Compute full-round S3 cost: (NM+M) PUTs + 2NM GETs."""
    puts = N * M + M
    gets = 2 * N * M
    return puts * 0.000005 + gets * 0.0000004

# ResNet-18 (45 MB), N=20, 10 rounds
RN18_LAMBDA = [0.000011, 0.000010, 0.000012, 0.000008, 0.000010]
RN18_S3 = [_s3_cost_per_round(N_CLIENTS, M) * 10 for M in SHARD_COUNTS]

# VGG-16 (528 MB), N=20, 10 rounds
VGG_LAMBDA = [0.001055, 0.001004, 0.000712, 0.000347, 0.000254]
VGG_S3 = [_s3_cost_per_round(N_CLIENTS, M) * 10 for M in SHARD_COUNTS]


def generate_cost_vs_model():
    """Generate fig_cost_vs_model: cost trajectories across model sizes."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # X positions (log scale)
    x = np.array(MODEL_SIZES)

    # Plot Lambda-FL (with gap for infeasible)
    lfl_x = [s for s, c in zip(MODEL_SIZES, LAMBDA_FL_COST) if c is not None]
    lfl_y = [c for c in LAMBDA_FL_COST if c is not None]
    ax.plot(lfl_x, lfl_y, 'o-', color='#1f77b4', linewidth=1.5, markersize=5,
            label=r'$\lambda$-FL', zorder=3)

    # Plot LIFL
    lifl_x = [s for s, c in zip(MODEL_SIZES, LIFL_COST) if c is not None]
    lifl_y = [c for c in LIFL_COST if c is not None]
    ax.plot(lifl_x, lifl_y, 's--', color='#2ca02c', linewidth=1.5, markersize=5,
            label='LIFL', zorder=3)

    # Plot GradsSharding (full range)
    ax.plot(MODEL_SIZES, GS_COST, 'D-', color='#d62728', linewidth=1.5, markersize=5,
            label='GradsSharding ($M$=4)', zorder=3)

    # Mark infeasible region
    ax.axvspan(5000, 15000, alpha=0.08, color='gray')
    ax.annotate('Infeasible for\ntree-based', xy=(7500, 50), fontsize=7,
                ha='center', color='gray', style='italic')

    # Mark crossover
    ax.axvline(x=500, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate('crossover\n~500 MB', xy=(500, 0.2), fontsize=6.5,
                ha='center', color='gray')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model gradient size (MB)')
    ax.set_ylabel('Total cost (USD / 1,000 rounds)')
    ax.set_xticks(MODEL_SIZES)
    ax.set_xticklabels(['45', '528', '2,048', '10,240'], fontsize=7)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(30, 15000)
    ax.set_ylim(0.15, 120)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_cost_vs_model.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig_cost_vs_model.png", dpi=300, bbox_inches='tight')
    print(f"Saved fig_cost_vs_model to {OUTPUT_DIR}")
    plt.close(fig)


def generate_cost_breakdown():
    """Generate fig_cost_breakdown: Lambda vs S3 cost for GradsSharding across M."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.4))

    x = np.arange(len(SHARD_COUNTS))
    width = 0.35

    # Left panel: ResNet-18
    ax1.bar(x - width/2, RN18_LAMBDA, width, label='Lambda', color='#1f77b4', zorder=3)
    ax1.bar(x + width/2, RN18_S3, width, label='S3 I/O', color='#ff7f0e', zorder=3)
    ax1.set_xlabel('Shard count $M$', fontsize=8)
    ax1.set_ylabel('Cost per experiment (USD)', fontsize=8)
    ax1.set_title('ResNet-18 (45 MB)', fontsize=8, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(SHARD_COUNTS, fontsize=7)
    ax1.set_yscale('log')
    ax1.legend(fontsize=6, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', which='both')

    # Right panel: VGG-16
    ax2.bar(x - width/2, VGG_LAMBDA, width, label='Lambda', color='#1f77b4', zorder=3)
    ax2.bar(x + width/2, VGG_S3, width, label='S3 I/O', color='#ff7f0e', zorder=3)
    ax2.set_xlabel('Shard count $M$', fontsize=8)
    ax2.set_title('VGG-16 (528 MB)', fontsize=8, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(SHARD_COUNTS, fontsize=7)
    ax2.set_yscale('log')
    ax2.legend(fontsize=6, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y', which='both')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_cost_breakdown.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig_cost_breakdown.png", dpi=300, bbox_inches='tight')
    print(f"Saved fig_cost_breakdown to {OUTPUT_DIR}")
    plt.close(fig)


if __name__ == "__main__":
    generate_cost_vs_model()
    generate_cost_breakdown()
    print("Done!")
