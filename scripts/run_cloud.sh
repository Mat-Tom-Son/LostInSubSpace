#!/bin/bash

# Cloud Setup Script for Othello-GPT A100 Run

echo "Setting up Othello-GPT Environment..."

# 1. Install dependencies
pip install -r requirements_cloud.txt

# 2. Check GPU
echo "Checking GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 3. Optimize for A100 (if detected)
# Set batch size to 2048 for A100 (80GB VRAM can handle ~4096 but 2048 is safe/fast)
BATCH_SIZE=2048

# 4. Run Experiment
echo "Starting Training (Batch Size: $BATCH_SIZE)..."
# We skip n_train_games arg to use default 100K unless specified
python experiments/phase_6/exp_o0_baseline.py --batch_size $BATCH_SIZE --lr 1e-3 --training_steps 25000

echo "Training Complete!"
