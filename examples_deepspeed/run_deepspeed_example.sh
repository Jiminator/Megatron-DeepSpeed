#!/bin/bash
set -ex

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
ZERO_STAGE=0

OUTPUT_DIR=output/PP${PP}_TP${TP}_DP${DP}_bs${MICRO_BATCH}
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
export MASTER_IP=$2
# export NCCL_P2P_LEVEL=NVL

ds_args=""
ds_args=" --deepspeed ${ds_args}"
# ds_args=" --no-pipeline-parallel ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"



deepspeed \
    --hostfile=./myhostfile \
    --no_ssh \
    --node_rank=$1 \
    --master_addr=${MASTER_IP} \
    --master_port=9875 \
    pretrain_gpt.py \
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
    --exit-interval 5000 | tee ${OUTPUT_DIR}/output.log