#!/usr/bin/env python3
"""Generate RQ3 cross-architecture comparison figure."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import numpy as np

# Data from experiments
models = ['ResNet-18\n(42.7 MB)', 'VGG-16\n(512 MB)', 'GPT-2 Large\n(2,953 MB)', 'Synthetic\n5 GB']

# Wall-clock times (mean ± std)
gs_time   = [7.9,   67.2,  362.5, 299.4]
gs_std    = [0.8,   5.1,   26.8,  9.5]
lfl_time  = [10.4,  117.3, None,  None]
lfl_std   = [0.8,   10.7,  None,  None]
lifl_time = [12.3,  126.6, None,  None]
lifl_std  = [1.1,   26.5,  None,  None]

# Cost per 1000 rounds (full round-trip S3: uploads + aggregation + read-back)
gs_cost   = [0.70,  3.82,  59.29, 85.66]
lfl_cost  = [0.38,  10.28, None,  None]
lifl_cost = [0.52,  13.03, None,  None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

x = np.arange(len(models))
width = 0.25

# --- Panel (a): Wall-clock time ---
bars1 = ax1.bar(x - width, gs_time, width, yerr=gs_std, label='GradsSharding',
                color='#2196F3', capsize=4, edgecolor='black', linewidth=0.5)

lfl_vals = [lfl_time[0], lfl_time[1], 0, 0]
lfl_errs = [lfl_std[0], lfl_std[1], 0, 0]
bars2 = ax1.bar(x[:2], lfl_vals[:2], width, yerr=lfl_errs[:2], label='λ-FL',
                color='#FF9800', capsize=4, edgecolor='black', linewidth=0.5)

lifl_vals = [lifl_time[0], lifl_time[1], 0, 0]
lifl_errs = [lifl_std[0], lifl_std[1], 0, 0]
bars3 = ax1.bar(x[:2] + width, lifl_vals[:2], width, yerr=lifl_errs[:2], label='LIFL',
                color='#4CAF50', capsize=4, edgecolor='black', linewidth=0.5)

# Mark infeasible
for i in [2, 3]:
    ax1.bar(i, 0, width, color='#FF9800', alpha=0.15, hatch='///', edgecolor='#FF9800', linewidth=0.5)
    ax1.bar(i + width, 0, width, color='#4CAF50', alpha=0.15, hatch='///', edgecolor='#4CAF50', linewidth=0.5)
    mid = i + width/2
    label = 'Not deployed' if i == 2 else 'Infeasible'
    ax1.text(mid, max(gs_time) * 0.12, label, ha='center', va='bottom',
             fontsize=12, fontstyle='italic', color='#666666', rotation=90)

ax1.set_ylabel('Wall-Clock Time (s)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_title('(a) Aggregation Latency', fontsize=15, fontweight='bold')
ax1.legend(fontsize=12, loc='upper left')
ax1.set_ylim(bottom=0)
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(axis='y', alpha=0.3)

# --- Panel (b): Cost per 1000 rounds ---
bars4 = ax2.bar(x - width, gs_cost, width, label='GradsSharding',
                color='#2196F3', edgecolor='black', linewidth=0.5)

lfl_cost_plot = [lfl_cost[0], lfl_cost[1]]
bars5 = ax2.bar(x[:2], lfl_cost_plot, width, label='λ-FL',
                color='#FF9800', edgecolor='black', linewidth=0.5)

lifl_cost_plot = [lifl_cost[0], lifl_cost[1]]
bars6 = ax2.bar(x[:2] + width, lifl_cost_plot, width, label='LIFL',
                color='#4CAF50', edgecolor='black', linewidth=0.5)

# Mark infeasible
for i in [2, 3]:
    ax2.bar(i, 0, width, color='#FF9800', alpha=0.15, hatch='///', edgecolor='#FF9800', linewidth=0.5)
    ax2.bar(i + width, 0, width, color='#4CAF50', alpha=0.15, hatch='///', edgecolor='#4CAF50', linewidth=0.5)
    mid = i + width/2
    label = 'Not deployed' if i == 2 else 'Infeasible'
    ax2.text(mid, max(gs_cost) * 0.12, label, ha='center', va='bottom',
             fontsize=12, fontstyle='italic', color='#666666', rotation=90)

# Add cost labels on top of GradsSharding bars for large models
for i in [2, 3]:
    ax2.text(i - width, gs_cost[i] + 2, f'${gs_cost[i]:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Cost per 1,000 Rounds ($)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=12)
ax2.set_title('(b) Cost Comparison', fontsize=15, fontweight='bold')
ax2.legend(fontsize=12, loc='upper left')
ax2.set_ylim(bottom=0)
ax2.tick_params(axis='y', labelsize=12)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/zealous-bold-gauss/mnt/grads-sharding/IC2E2026-ServerlessScalability/images/fig_rq3_comparison.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/sessions/zealous-bold-gauss/mnt/grads-sharding/IC2E2026-ServerlessScalability/images/fig_rq3_comparison.png',
            bbox_inches='tight', dpi=150)
print("Figure saved successfully.")
