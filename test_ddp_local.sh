#!/bin/bash
# Local testing script for DDP training (for development/debugging)
# This script can be used to test DDP functionality on a local machine with multiple GPUs

echo "Starting local DDP training test..."

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

# Test with small parameters for quick validation
python src/fdslxsdf4seg/training_ddp.py \
    --data_json_path BTCV/dataset.json \
    --model_name vnet \
    --batch_size 1 \
    --max_iterations 100 \
    --learning_rate 1e-4 \
    --out_channel 14 \
    --grid_size 64 64 64 \
    --out_dir test_ddp_output \
    --is_real_data

echo "Local DDP training test completed!"
echo "Check test_ddp_output/ for results"
