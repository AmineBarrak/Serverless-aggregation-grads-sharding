#!/bin/bash
# =============================================================================
# Matilda HPC - One-time environment setup
# Run this ONCE after cloning the repo to /home/a/aminebarrak/grads-sharding
# =============================================================================

set -e

echo "========================================="
echo "Setting up GradsSharding on Matilda HPC"
echo "========================================="

# Load modules
module load Python/3.10.14
module load CUDA/11.8
module load cuDNN/8.9.4-cuda11.8

echo "[1/3] Creating Python virtual environment..."
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
echo "Repo directory: $REPO_DIR"

python3 -m venv venv
source venv/bin/activate

echo "[2/3] Installing dependencies..."
pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib seaborn

echo "[3/3] Verifying installation..."
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'Memory:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import matplotlib
print(f'Matplotlib: {matplotlib.__version__}')
print('All good!')
"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate in future sessions:"
echo "  module load Python/3.10.14 CUDA/11.8 cuDNN/8.9.4-cuda11.8"
echo "  source $REPO_DIR/venv/bin/activate"
