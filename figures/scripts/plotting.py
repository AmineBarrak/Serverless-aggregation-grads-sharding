"""
Plotting functions for experiment results.

Generates publication-ready figures for all 4 Research Questions.
Uses matplotlib + seaborn with professional styling.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC/headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Set professional style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"

# Color scheme: Lambda-FL=blue, LIFL=green, GradsSharding=red/orange
COLORS = {
    "lambda-fl": "#1f77b4",      # Blue
    "lifl": "#2ca02c",            # Green
    "grads-sharding": "#d62728",  # Red
}

SYSTEM_NAMES = {
    "lambda-fl": "Lambda-FL",
    "lifl": "LIFL",
    "grads-sharding": "GradsSharding",
}


def plot_rq1(results_dir: Path) -> None:
    """
    Plot RQ1 - Scalability with client count.

    Creates two subplots:
    - Left: Latency (line per system) vs client count
    - Right: Cost (grouped bars) vs client count
    """
    results_file = results_dir / "rq1_scalability" / "rq1_results.json"
    if not results_file.exists():
        print(f"Warning: RQ1 results not found at {results_file}")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract data - JSON keys are strings
    client_counts = sorted([int(k) for k in results.keys() if k.isdigit()])
    systems = ["lambda-fl", "lifl", "grads-sharding"]

    latencies = {sys: [] for sys in systems}
    costs = {sys: [] for sys in systems}

    for nc in client_counts:
        nc_str = str(nc)
        for system in systems:
            data = results.get(nc_str, {}).get(system, {})
            latencies[system].append(data.get("avg_round_latency_s", data.get("elapsed_time_s", 0)))
            costs[system].append(data.get("total_cost_usd", 0))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Latency
    for system in systems:
        ax1.plot(
            client_counts,
            latencies[system],
            marker="o",
            linewidth=2,
            label=SYSTEM_NAMES[system],
            color=COLORS[system],
        )

    ax1.set_xlabel("Number of Clients", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Round Latency (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("RQ1: Scalability - Latency vs Client Count", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cost
    x_pos = np.arange(len(client_counts))
    bar_width = 0.25

    for i, system in enumerate(systems):
        ax2.bar(
            x_pos + i * bar_width,
            costs[system],
            bar_width,
            label=SYSTEM_NAMES[system],
            color=COLORS[system],
            alpha=0.8,
        )

    ax2.set_xlabel("Number of Clients", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Cost (USD)", fontsize=11, fontweight="bold")
    ax2.set_title("RQ1: Scalability - Cost vs Client Count", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos + bar_width)
    ax2.set_xticklabels(client_counts)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "experiments" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / "rq1_scalability.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "rq1_scalability.pdf", bbox_inches="tight")
    print(f"Saved RQ1 plots to {figures_dir}/rq1_scalability.*")
    plt.close(fig)


def plot_rq2(results_dir: Path) -> None:
    """
    Plot RQ2 - Shard ablation study.

    Creates three subplots:
    - Left: Memory vs shard count
    - Middle: Latency vs shard count
    - Right: Cold start ratio vs shard count
    """
    results_file = results_dir / "rq2_shard_ablation" / "rq2_results.json"
    if not results_file.exists():
        print(f"Warning: RQ2 results not found at {results_file}")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract data - keys are "shards_1", "shards_2", "shards_4", etc.
    shard_counts = sorted([
        int(k.replace("shards_", ""))
        for k in results.keys()
        if k.startswith("shards_")
    ])

    memory_values = []
    latency_values = []
    lifl_latency = None

    # Get LIFL baseline latency if available
    if "lifl_baseline" in results:
        lifl_latency = results["lifl_baseline"].get(
            "avg_round_latency_s",
            results["lifl_baseline"].get("elapsed_time_s", 0)
        )

    for num_shards in shard_counts:
        data = results.get(f"shards_{num_shards}", {})
        latency_values.append(data.get("avg_round_latency_s", data.get("elapsed_time_s", 0)))
        memory_values.append(data.get("avg_peak_memory_mb", 0))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Memory vs shard count
    ax1.plot(
        shard_counts,
        memory_values,
        marker="s",
        linewidth=2,
        markersize=8,
        color=COLORS["grads-sharding"],
        label="GradsSharding",
    )
    ax1.set_xlabel("Number of Shards", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Avg Peak Memory (MB)", fontsize=11, fontweight="bold")
    ax1.set_title("RQ2: Shard Ablation - Memory Usage", fontsize=12, fontweight="bold")
    ax1.set_xticks(shard_counts)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Latency vs shard count (with LIFL baseline)
    ax2.plot(
        shard_counts,
        latency_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color=COLORS["grads-sharding"],
        label="GradsSharding",
    )
    if lifl_latency is not None:
        ax2.axhline(
            y=lifl_latency, color=COLORS["lifl"], linestyle="--",
            linewidth=2, label="LIFL Baseline",
        )
    ax2.set_xlabel("Number of Shards", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Avg Round Latency (seconds)", fontsize=11, fontweight="bold")
    ax2.set_title("RQ2: Shard Ablation - Latency", fontsize=12, fontweight="bold")
    ax2.set_xticks(shard_counts)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "experiments" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / "rq2_shard_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "rq2_shard_ablation.pdf", bbox_inches="tight")
    print(f"Saved RQ2 plots to {figures_dir}/rq2_shard_ablation.*")
    plt.close(fig)


def plot_rq3(results_dir: Path) -> None:
    """
    Plot RQ3 - Head-to-head comparison.

    Creates two subplots:
    - Left: Time-to-accuracy and cost grouped bars
    - Right: Average round latency table
    """
    results_file = results_dir / "rq3_headtohead" / "rq3_results.json"
    if not results_file.exists():
        print(f"Warning: RQ3 results not found at {results_file}")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract data - keys are directly "lambda-fl", "lifl", "grads-sharding"
    systems = ["lambda-fl", "lifl", "grads-sharding"]
    elapsed_times = []
    costs = []
    avg_latencies = []

    for system in systems:
        if system in results:
            data = results[system]
            elapsed_times.append(data.get("elapsed_time_s", 0))
            costs.append(data.get("total_cost_usd", 0))
            avg_latencies.append(data.get("avg_round_latency_s", 0))
        else:
            elapsed_times.append(0)
            costs.append(0)
            avg_latencies.append(0)

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # Create GridSpec for different sized subplots
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # Plot 1: Time to target accuracy
    x_pos = np.arange(len(systems))
    ax1.bar(
        x_pos,
        elapsed_times,
        color=[COLORS[s] for s in systems],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("RQ3: Training Time to Convergence", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([SYSTEM_NAMES[s] for s in systems])
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Cost
    ax2.bar(
        x_pos,
        costs,
        color=[COLORS[s] for s in systems],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.set_ylabel("Cost (USD)", fontsize=11, fontweight="bold")
    ax2.set_title("RQ3: Total Cost", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([SYSTEM_NAMES[s] for s in systems])
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Average latency table
    ax3.axis("off")

    table_data = [
        ["System", "Avg Round\nLatency (s)"],
    ]
    for i, system in enumerate(systems):
        table_data.append([SYSTEM_NAMES[system], f"{avg_latencies[i]:.2f}"])

    table = ax3.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.6, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
            table[(i, j)].set_text_props(weight="bold")

    ax3.set_title("RQ3: Average Round Latency", fontsize=11, fontweight="bold", pad=20)

    plt.suptitle("RQ3: Head-to-Head Comparison", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "experiments" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / "rq3_headtohead.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "rq3_headtohead.pdf", bbox_inches="tight")
    print(f"Saved RQ3 plots to {figures_dir}/rq3_headtohead.*")
    plt.close(fig)


def plot_rq4(results_dir: Path) -> None:
    """
    Plot RQ4 - Cost breakdown of GradsSharding.

    Creates two subplots:
    - Left: Pie chart of cost distribution
    - Right: Stacked bar showing cost components
    """
    results_file = results_dir / "rq4_cost_breakdown" / "rq4_results.json"
    if not results_file.exists():
        print(f"Warning: RQ4 results not found at {results_file}")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract cost breakdown - nested under results["grads-sharding"]["cost_breakdown"]
    gs_data = results.get("grads-sharding", {})
    cost_breakdown = gs_data.get("cost_breakdown", {})
    if not cost_breakdown:
        print("Warning: Cost breakdown not available in RQ4 results")
        return

    # Fields from cost_model.py: lambda_cost_usd, s3_put_cost_usd, s3_get_cost_usd,
    # s3_storage_cost_usd, step_function_cost_usd, total_cost_usd
    lambda_cost = cost_breakdown.get("lambda_cost_usd", 0)
    s3_cost = (
        cost_breakdown.get("s3_put_cost_usd", 0)
        + cost_breakdown.get("s3_get_cost_usd", 0)
        + cost_breakdown.get("s3_storage_cost_usd", 0)
    )
    step_fn_cost = cost_breakdown.get("step_function_cost_usd", 0)
    total_cost = cost_breakdown.get("total_cost_usd", lambda_cost + s3_cost + step_fn_cost)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pie chart
    components = ["Lambda Compute", "S3 Storage", "Step Functions"]
    costs = [lambda_cost, s3_cost, step_fn_cost]
    colors_pie = ["#1f77b4", "#2ca02c", "#d62728"]

    # Handle all-zero case (avoid matplotlib warning)
    if sum(costs) == 0:
        costs = [1, 1, 1]  # equal placeholder

    wedges, texts, autotexts = ax1.pie(
        costs,
        labels=components,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10, "weight": "bold"},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")

    ax1.set_title("RQ4: Cost Distribution by Component", fontsize=12, fontweight="bold")

    # Plot 2: Stacked bar
    x_pos = [0]
    ax2.barh(x_pos, [lambda_cost], color="#1f77b4", label="Lambda Compute")
    ax2.barh(x_pos, [s3_cost], left=[lambda_cost], color="#2ca02c", label="S3 Storage")
    ax2.barh(
        x_pos, [step_fn_cost],
        left=[lambda_cost + s3_cost],
        color="#d62728", label="Step Functions",
    )

    ax2.set_xlabel("Cost (USD)", fontsize=11, fontweight="bold")
    ax2.set_title(f"RQ4: Cost Breakdown (Total: ${total_cost:.6f})", fontsize=12, fontweight="bold")
    ax2.set_yticks([])
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    # Save figure
    figures_dir = Path(__file__).parent.parent / "experiments" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / "rq4_cost_breakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "rq4_cost_breakdown.pdf", bbox_inches="tight")
    print(f"Saved RQ4 plots to {figures_dir}/rq4_cost_breakdown.*")
    plt.close(fig)


def plot_all(results_dir: Path) -> None:
    """Generate all plots."""
    print("\nGenerating plots...")
    plot_rq1(results_dir)
    plot_rq2(results_dir)
    plot_rq3(results_dir)
    plot_rq4(results_dir)
    print("All plots generated successfully!")


if __name__ == "__main__":
    results_dir = PROJECT_ROOT / "experiments" / "results"
    plot_all(results_dir)
