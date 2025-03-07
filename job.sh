#!/bin/bash
#SBATCH --account=bcrn-delta-gpu
#SBATCH --job-name=megatron-deepspeed
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # DO NOT CHANGE
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=output/output_%j.out
#SBATCH --error=output/output_%j.err

set -ex

export LOGLEVEL=INFO

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate llm
echo "START TIME: $(date)"

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load nccl # loads the nccl built with the AWS nccl plugin for Slingshot11
module list
echo "Job is starting on `hostname`"

BASE_PATH=../oscar
DATA_PATH=../meg-gpt2_text_document
DS_CONFIG=ds_config.json

WORLD_SIZE=$SLURM_NTASKS
TP=2
PP=1
DP=$((WORLD_SIZE / (TP * PP)))

NLAYERS=24
HIDDEN=512

GLOBAL_BATCH=64
MICRO_BATCH=64

OUTPUT_DIR=output/${SLURM_JOB_ID}/PP${PP}_TP${TP}_DP${DP}_bs${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

# TODO Figure out how to get fp16 working.
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "fp16": {
    "enabled": false,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true,

  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": "${OUTPUT_DIR}/MULTINODE_PROFILER.txt"
  }
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
# ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NNODES=$SLURM_NNODES

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

bash makehostfile.sh

export LAUNCHER="deepspeed \
    --hostfile=./myhostfile \
    --no_ssh \
    --master_addr=${MASTER_ADDR} \
    --master_port=$MASTER_PORT \
    "

CMD="${SLURM_SUBMIT_DIR}/pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads 16 \
    --seq-length 256 \
    --loss-scale 12 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 2 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 2 \
    --eval-interval 1000 \
    --data-path $DATA_PATH \
    --vocab-file $BASE_PATH/gpt2-vocab.json \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --tensorboard-dir $OUTPUT_DIR \
    $ds_args \
    --exit-interval 5000
    "

echo $CMD

clear; srun --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD"

conda deactivate