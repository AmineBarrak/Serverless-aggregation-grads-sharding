#!/bin/bash
#SBATCH --job-name=fl-all-rqs
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/all_rqs_%j.out
#SBATCH --error=logs/all_rqs_%j.err
# =============================================================================
# Run ALL 2 RQs sequentially in one job
# Full scale: ~10-12 hours total
# Use this if you prefer one job instead of 2 parallel ones
# =============================================================================

set -e

module load Python/3.10.14
module load CUDA/11.8
module load cuDNN/8.9.4-cuda11.8

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
mkdir -p logs

echo "================================================"
echo "ALL RQs: Full Experiment Suite - Job $SLURM_JOB_ID"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "================================================"

python experiments/run_experiment.py --rq all

echo ""
echo "Completed: $(date)"
echo "All results: experiments/results/"
echo "All figures: experiments/figures/"
