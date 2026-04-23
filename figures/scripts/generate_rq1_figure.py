"""
Generate the RQ1 figure: PS Idle Time across model sizes.
Stacked bar chart showing Worker Compute vs PS Aggregation time,
with PS idle percentage annotated on each bar.

Run after rq1_ps_idle_time.py has produced results.
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

RESULTS_FILE = Path(__file__).parent / "results" / "rq1_ps_idle_time" / "rq1_ps_idle_results.json"

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


def generate_from_results(results_file):
    """Generate figure from actual experimental results."""
    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]
    n_clients = data.get("fl_config", data.get("config", {})).get("num_clients_N",
                data.get("fl_config", data.get("config", {})).get("num_clients", 20))

    model_names = []
    client_times = []
    agg_times = []
    idle_pcts = []
    param_counts = []

    for r in results:
        model_names.append(r["model_name"])
        rt = r["round_timing"]
        # Support both old and new JSON schema
        client_times.append(rt.get("t_train_ms", rt.get("parallel_client_ms", 0)))
        agg_times.append(rt.get("t_agg_ms", rt.get("aggregation_ms", 0)))
        idle_pcts.append(rt.get("ps_idle_pct", rt.get("ps_idle_pct_parallel", 0)))
        param_counts.append(r["num_parameters"])

    _plot_figure(model_names, param_counts, client_times, agg_times, idle_pcts, n_clients)


def _plot_figure(model_names, param_counts, client_times, agg_times, idle_pcts, n_clients):
    """Create the stacked bar chart."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(model_names))
    width = 0.5

    # Stacked bars: worker compute (bottom) + PS aggregation (top)
    bars_client = ax.bar(x, client_times, width, label='Client Training (GPU)',
                         color='#4A90D9', zorder=3, edgecolor='white', linewidth=0.5)
    bars_agg = ax.bar(x, agg_times, width, bottom=client_times,
                      label='PS Aggregation (CPU)',
                      color='#F5A623', zorder=3, edgecolor='white', linewidth=0.5)

    # Annotate total time on top of each bar
    for i, (ct, at, idle) in enumerate(zip(client_times, agg_times, idle_pcts)):
        total = ct + at
        # Total time label above bar
        ax.text(i, total * 1.15, f"{total/1000:.1f}s",
                ha='center', va='bottom', fontsize=7, fontweight='bold')
        # PS idle percentage inside the worker bar
        ax.text(i, ct * 0.5, f"{idle:.1f}%\nidle",
                ha='center', va='center', fontsize=6.5, fontweight='bold',
                color='white')

    # X-axis labels with param count
    x_labels = []
    for name, params in zip(model_names, param_counts):
        if params >= 1e9:
            p_str = f"{params/1e9:.1f}B params"
        else:
            p_str = f"{params/1e6:.0f}M params"
        x_labels.append(f"{name}\n{p_str}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel('Round Time (ms, log scale)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.3, axis='y', which='both')

    # Set y-axis range with headroom for labels
    all_totals = [c + a for c, a in zip(client_times, agg_times)]
    ax.set_ylim(min(client_times) * 0.3, max(all_totals) * 3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_ps_idle_time.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig_ps_idle_time.png", dpi=300, bbox_inches='tight')
    print(f"Saved fig_ps_idle_time to {OUTPUT_DIR}")
    plt.close(fig)


if __name__ == "__main__":
    if RESULTS_FILE.exists():
        print(f"Loading results from {RESULTS_FILE}")
        generate_from_results(RESULTS_FILE)
    else:
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run rq1_ps_idle_time.py first, then run this script.")
        sys.exit(1)
