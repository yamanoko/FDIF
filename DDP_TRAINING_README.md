# Distributed Data Parallel (DDP) Training for ABCI

This document explains how to use the DDP training functionality for both single-node and multi-node training on ABCI 3.0.

## Overview

The new DDP training script (`training_ddp.py`) enables distributed training across multiple GPUs and nodes, providing significant speedup compared to single-GPU training. This implementation is based on the ABCI DDP training guidelines.

## Files Added

- `src/fdslxsdf4seg/training_ddp.py`: Main DDP training script
- `run_ddp_single_node.sh`: PBS job script for single-node DDP training (8 GPUs)
- `run_ddp_multi_node.sh`: PBS job script for multi-node DDP training (multiple nodes × 8 GPUs)
- `DDP_TRAINING_README.md`: This documentation file

## Key Features

### DDP Training Script (`training_ddp.py`)

- **Automatic Mode Detection**: Automatically detects whether to run in single-node or multi-node mode based on environment variables
- **Distributed Data Loading**: Uses `DistributedSampler` for efficient data distribution across GPUs
- **Memory Optimization**: Includes optimizations for GPU memory management in distributed settings
- **Checkpointing**: Supports resuming training from checkpoints in distributed environment
- **Synchronized Validation**: Performs validation across all GPUs with synchronized metrics
- **Rank 0 Only Operations**: File I/O, logging, and model saving are performed only by rank 0 to avoid conflicts

### Single-Node DDP

- Uses all 8 GPUs on a single rt_HF node
- Communication happens within the node (faster)
- Uses `torch.multiprocessing.spawn` for process management

### Multi-Node DDP

- Uses multiple rt_HF nodes (each with 8 GPUs)
- Requires MPI for inter-node communication
- Uses NCCL backend for efficient GPU-to-GPU communication
- Master node coordinates the training

## Usage

### Prerequisites

1. Ensure you have a virtual environment set up with all required packages
2. Update the `[group_name]` placeholder in the PBS scripts with your ABCI group name
3. Adjust paths and parameters as needed

### Single-Node DDP Training

```bash
# Submit single-node DDP job (8 GPUs)
qsub run_ddp_single_node.sh
```

The single-node script will automatically use all 8 GPUs on the allocated node.

### Multi-Node DDP Training

```bash
# Submit multi-node DDP job (4 nodes × 8 GPUs = 32 GPUs)
qsub run_ddp_multi_node.sh
```

To change the number of nodes, modify the `select=4` parameter in `run_ddp_multi_node.sh`.

### Command Line Arguments

The DDP training script supports all the same arguments as the original `training.py`:

```bash
python src/fdslxsdf4seg/training_ddp.py \
    --data_json_path BTCV/dataset.json \
    --model_name vnet \
    --batch_size 2 \
    --max_iterations 10000 \
    --learning_rate 1e-4 \
    --out_channel 14 \
    --out_dir output_directory \
    --is_real_data \
    --pretrained_model path/to/pretrained.pth \
    --resume_from_checkpoint path/to/checkpoint.pth
```

### Important Considerations

#### Batch Size

When using DDP, the `--batch_size` parameter specifies the batch size **per GPU**. The effective global batch size will be:

```
Global Batch Size = batch_size × number_of_GPUs
```

For example:
- Single-node (8 GPUs) with `--batch_size 2` → Global batch size = 16
- Multi-node (4 nodes × 8 GPUs) with `--batch_size 2` → Global batch size = 64

#### Learning Rate Scaling

When increasing the global batch size, you may need to scale the learning rate accordingly. A common approach is linear scaling:

```
new_learning_rate = base_learning_rate × (global_batch_size / base_batch_size)
```

#### Resource Requirements

- **Single-node**: Uses 1 rt_HF node (8 H200 GPUs)
- **Multi-node**: Uses multiple rt_HF nodes (e.g., 4 nodes = 32 H200 GPUs)

## Performance Expectations

Based on typical DDP scaling:

- **Single-node DDP**: ~7-8x speedup compared to single GPU
- **Multi-node DDP**: Near-linear scaling with slight communication overhead

The actual speedup depends on:
- Model size and complexity
- Batch size
- Network communication overhead (for multi-node)
- Data loading efficiency

## Output Files

DDP training produces the same output files as single-GPU training:

- `best_metric_model.pth`: Best performing model
- `last_model.pth`: Latest model from final validation
- `training_checkpoint.pth`: Training checkpoint for resuming
- Training metrics and visualizations
- Log files with per-class dice scores

## Troubleshooting

### Common Issues

1. **NCCL Communication Errors**: Ensure proper network configuration and firewall settings
2. **Memory Issues**: Reduce batch size per GPU if running into OOM errors
3. **Synchronization Issues**: All processes must reach barriers at the same time

### Debugging

- Check the output logs for each rank
- Verify that all processes are properly initialized
- Monitor GPU memory usage across all nodes

### Environment Variables (Multi-Node)

The script automatically detects multi-node setup using these environment variables:
- `OMPI_COMM_WORLD_RANK`: Global rank of the process
- `OMPI_COMM_WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Address of the master node
- `MASTER_PORT`: Port for communication

## Comparison with Original Training

| Feature | Original `training.py` | DDP `training_ddp.py` |
|---------|----------------------|----------------------|
| GPU Usage | Single GPU | Multiple GPUs/Nodes |
| Training Speed | Baseline | 7-8x faster (single-node), >20x faster (multi-node) |
| Memory Usage | Single GPU memory | Distributed across GPUs |
| Complexity | Simple | More complex setup |
| Resource Cost | Lower | Higher (more GPUs) |

## Best Practices

1. **Start with Single-Node**: Test your configuration with single-node DDP before moving to multi-node
2. **Monitor Training**: Keep an eye on loss convergence and validation metrics
3. **Adjust Batch Size**: Scale batch size appropriately for the number of GPUs
4. **Use Checkpointing**: Regularly save checkpoints for long training runs
5. **Resource Planning**: Consider the trade-off between speed and resource cost

## Example Workflows

### Development and Testing
1. Use original `training.py` for quick experiments and debugging
2. Use single-node DDP for faster training during development

### Production Training
1. Use multi-node DDP for large-scale training runs
2. Adjust learning rate and batch size for optimal convergence
3. Monitor resource usage and training progress

This DDP implementation provides a scalable solution for training large medical segmentation models efficiently on ABCI's high-performance computing infrastructure.
