#!/bin/bash
#SBATCH --job-name=fl-rq2-arch
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq2_%j.out
#SBATCH --error=logs/rq2_%j.err
# =============================================================================
# RQ2: Cross-Architecture Cost & Scalability Analysis
# Part A (HPC): All 3 systems on VGG-16, N=50, 20 rounds (~5 hours)
# Part B (Analytical): Cost projection, no GPU needed (~1 minute)
# =============================================================================

set -e

module load Python/3.10.14
module load CUDA/11.8
module load cuDNN/8.9.4-cuda11.8

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
mkdir -p logs

echo "================================================"
echo "RQ2: Cross-Architecture - Job $SLURM_JOB_ID"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "================================================"

# Run both Part A (HPC measured) and Part B (analytical)
python experiments/run_experiment.py --rq 2 --part both

echo ""
echo "Completed: $(date)"
echo "Results: experiments/results/rq2_cross_architecture/"
