#!/bin/sh
#PBS -q rt_HF
#PBS -l select=4:mpiprocs=192
#PBS -l walltime=24:00:00
#PBS -P [group_name]

cd ${PBS_O_WORKDIR}
export PYTHONPATH=${PBS_O_WORKDIR}:$PYTHONPATH

export RESULTS_DIR="training_output/ddp_multi_node/${PBS_JOBID}"
mkdir -p ${RESULTS_DIR}

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1 cudnn/9.5/9.5.1
module load hpcx/2.20

# Activate your virtual environment
# source venv/bin/activate

# Calculate number of processes (8 GPUs per node)
export NUM_NODES=4
export NUM_PROCESSES=$((NUM_NODES * 8))

# Set master node information for multi-node communication
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $NUM_NODES"
echo "Number of processes: $NUM_PROCESSES"

# Run multi-node DDP training with MPI
mpirun -np ${NUM_PROCESSES} -map-by ppr:8:node -hostfile $PBS_NODEFILE \
    python src/fdslxsdf4seg/training_ddp.py \
    --data_json_path BTCV/dataset.json \
    --model_name vnet \
    --batch_size 2 \
    --max_iterations 10000 \
    --learning_rate 1e-4 \
    --out_channel 14 \
    --out_dir ${RESULTS_DIR} \
    --is_real_data \
    2>&1 | tee ${RESULTS_DIR}/output.txt

echo "Multi-node DDP training completed!"
