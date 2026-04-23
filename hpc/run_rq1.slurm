#!/bin/bash
#SBATCH --job-name=fl-rq1-shard
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq1_%j.out
#SBATCH --error=logs/rq1_%j.err
# =============================================================================
# RQ1: Shard Ablation Study — Optimal K Across Model Scales
# Tests M in {1,2,4,8,16} for ResNet-18 and VGG-16
# N=20 clients, 10 rounds each configuration
# Estimated: ~4-5 hours (VGG-16 with M=1 is the bottleneck)
# =============================================================================

set -e

module load Python/3.10.14
module load CUDA/11.8
module load cuDNN/8.9.4-cuda11.8

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
mkdir -p logs

echo "================================================"
echo "RQ1: Shard Ablation - Job $SLURM_JOB_ID"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "================================================"

python experiments/run_experiment.py --rq 1

echo ""
echo "Completed: $(date)"
echo "Results: experiments/results/rq1_shard_ablation/"
