#!/bin/bash
#SBATCH --job-name=fl-rq3-h2h
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq3_%j.out
#SBATCH --error=logs/rq3_%j.err
# =============================================================================
# RQ3: Head-to-Head Comparison (B-Baseline)
# Full scale: RVL-CDIP, VGG-16, 50 clients, 3 systems, 20 rounds
# Estimated: ~5-6 hours (VGG-16 is large, needs more memory)
# =============================================================================

set -e

module load Python/3.10.14
module load CUDA/11.8
module load cuDNN/8.9.4-cuda11.8

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
mkdir -p logs

echo "================================================"
echo "RQ3: Head-to-Head - Job $SLURM_JOB_ID"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "================================================"

python -m experiments.rq3_headtohead

echo ""
echo "Completed: $(date)"
echo "Results: experiments/results/rq3_headtohead/"
