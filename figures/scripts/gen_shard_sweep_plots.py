import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = {}
for m in [1, 2, 4, 8, 16]:
    with open(f'/sessions/zealous-bold-gauss/mnt/grads-sharding/experiments/lambda_validation/results/vgg16_shard_sweep/vgg16_m{m}.json') as f:
        results[m] = json.load(f)

Ms = [1, 2, 4, 8, 16]
s3_read = [results[m]['s3_read_ms']['mean']/1000 for m in Ms]
compute = [results[m]['compute_ms']['mean']/1000 for m in Ms]
s3_write = [results[m]['s3_write_ms']['mean']/1000 for m in Ms]
agg_total = [results[m]['aggregation_ms']['mean']/1000 for m in Ms]
agg_std = [results[m]['aggregation_ms']['std']/1000 for m in Ms]
wall = [results[m]['wall_clock_ms']['mean']/1000 for m in Ms]

lambda_cost = [results[m]['cost_per_round']['lambda_compute']*1000 for m in Ms]
s3_cost = [results[m]['cost_per_round']['s3_io']*1000 for m in Ms]
total_cost = [results[m]['cost_per_round']['total']*1000 for m in Ms]
s3_throughput = [results[m]['s3_throughput_mbps'] for m in Ms]
s3_ops = [results[m]['s3_ops_per_round'] for m in Ms]
shard_mb = [results[m]['shard_mb'] for m in Ms]

# ============ PLOT 1: Stacked bar - Time breakdown ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

x = np.arange(len(Ms))
width = 0.55

bars1 = ax1.bar(x, s3_read, width, label='S3 Read', color='#2196F3')
bars2 = ax1.bar(x, compute, width, bottom=s3_read, label='FedAvg Compute', color='#FF9800')
bars3 = ax1.bar(x, s3_write, width, bottom=[s+c for s,c in zip(s3_read, compute)], label='S3 Write', color='#4CAF50')

# Add error bars for total aggregation time
ax1.errorbar(x, agg_total, yerr=agg_std, fmt='none', ecolor='black', capsize=4, linewidth=1.5)

ax1.set_xlabel('Number of Shards (M)', fontsize=11)
ax1.set_ylabel('Aggregation Time (s)', fontsize=11)
ax1.set_title('(a) Time Breakdown by Shard Count', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([str(m) for m in Ms])
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Add speedup annotations
for i in range(len(Ms)):
    speedup = agg_total[0] / agg_total[i]
    if i > 0:
        ax1.annotate(f'{speedup:.1f}×', xy=(x[i], agg_total[i] + agg_std[i] + 3),
                    ha='center', fontsize=9, fontweight='bold', color='#D32F2F')

# ============ PLOT 2: Cost breakdown ============
bars4 = ax2.bar(x, lambda_cost, width, label='Lambda Compute', color='#9C27B0')
bars5 = ax2.bar(x, s3_cost, width, bottom=lambda_cost, label='S3 I/O', color='#00BCD4')

ax2.set_xlabel('Number of Shards (M)', fontsize=11)
ax2.set_ylabel('Cost per 1K Rounds (USD)', fontsize=11)
ax2.set_title('(b) Cost Breakdown by Shard Count', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([str(m) for m in Ms])
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

# Add cost labels on bars
for i in range(len(Ms)):
    ax2.annotate(f'${total_cost[i]:.2f}', xy=(x[i], total_cost[i] + 0.15),
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/sessions/zealous-bold-gauss/mnt/grads-sharding/IC2E2026-ServerlessScalability/images/fig_vgg16_shard_sweep.pdf', 
            bbox_inches='tight', dpi=300)
plt.savefig('/sessions/zealous-bold-gauss/fig_vgg16_shard_sweep.png', 
            bbox_inches='tight', dpi=150)
print("Plot 1 saved: fig_vgg16_shard_sweep.pdf")

# ============ PLOT 3: Latency vs Cost tradeoff + S3 throughput ============
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4.5))

# Latency vs Cost scatter
ax3.plot(agg_total, total_cost, 'o-', color='#D32F2F', markersize=10, linewidth=2, zorder=5)
for i, m in enumerate(Ms):
    ax3.annotate(f'M={m}', xy=(agg_total[i], total_cost[i]),
                xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold')
ax3.set_xlabel('Aggregation Latency (s)', fontsize=11)
ax3.set_ylabel('Cost per 1K Rounds (USD)', fontsize=11)
ax3.set_title('(a) Latency vs. Cost Tradeoff', fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3)

# S3 ops and throughput
color1 = '#2196F3'
color2 = '#FF9800'
ax4_twin = ax4.twinx()

line1 = ax4.bar(x - 0.15, s3_ops, 0.3, color=color1, alpha=0.7, label='S3 Ops/Round')
line2 = ax4_twin.plot(x, s3_throughput, 'o-', color=color2, linewidth=2, markersize=8, label='S3 Throughput')

ax4.set_xlabel('Number of Shards (M)', fontsize=11)
ax4.set_ylabel('S3 Operations per Round', fontsize=11, color=color1)
ax4_twin.set_ylabel('S3 Throughput (MB/s)', fontsize=11, color=color2)
ax4.set_title('(b) S3 Operations and Throughput', fontsize=11, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([str(m) for m in Ms])
ax4.tick_params(axis='y', labelcolor=color1)
ax4_twin.tick_params(axis='y', labelcolor=color2)
ax4_twin.set_ylim(30, 70)
ax4.grid(axis='y', alpha=0.3)

# Combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('/sessions/zealous-bold-gauss/mnt/grads-sharding/IC2E2026-ServerlessScalability/images/fig_vgg16_shard_tradeoff.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/sessions/zealous-bold-gauss/fig_vgg16_shard_tradeoff.png',
            bbox_inches='tight', dpi=150)
print("Plot 2 saved: fig_vgg16_shard_tradeoff.pdf")

# Print summary table for paper
print("\n=== TABLE FOR PAPER ===")
print(f"{'M':>3} {'Shard':>8} {'Agg(s)':>8} {'S3 Read':>8} {'Compute':>8} {'S3 Write':>9} {'S3 Ops':>7} {'MB/s':>6} {'$/1K':>7} {'Speedup':>8}")
for i, m in enumerate(Ms):
    speedup = agg_total[0] / agg_total[i]
    print(f"{m:>3} {shard_mb[i]:>7.1f} {agg_total[i]:>8.1f} {s3_read[i]:>8.1f} {compute[i]:>8.2f} {s3_write[i]:>9.1f} {s3_ops[i]:>7} {s3_throughput[i]:>6.1f} {total_cost[i]:>7.2f} {speedup:>8.1f}x")

# Percentage of time spent on S3
print("\n=== S3 READ AS % OF TOTAL ===")
for i, m in enumerate(Ms):
    pct = s3_read[i] / agg_total[i] * 100
    print(f"M={m}: {pct:.1f}%")

