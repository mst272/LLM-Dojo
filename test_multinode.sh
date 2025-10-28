#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh && conda activate sft
conda info -e


# 启动DDP优化：llamafactory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export TORCH_DISTRIBUTED_RENDEZVOUS_BACKEND=etcd   # 不然加入节点会报错


export SWANLAB_API_KEY='api_key'




export FORCE_TORCHRUN=1
export NNODES=$PET_NNODES
export NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export PET_NPROC_PER_NODE=$PET_NPROC_PER_NODE


echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE}"




DATA_PATH=""
MODEL_PATH=""

swanlab_run_name=''
logfile=''
OUTPUT_PATH=""



torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main_train.py \
    --train_data_path "$DATA_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --max_len 16384 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --task_type "sft" \
    --train_mode "full" \
    --output_dir "$OUTPUT_PATH" \
    --save_strategy "epoch" \
    --save_steps 30000 \
    --save_total_limit 18 \
    --learning_rate 2e-5 \
    --router_aux_loss_coef 0.0 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "swanlab" \
    --run_name $swanlab_run_name \
    --bf16 True \
    --ddp_timeout 8400 \
    --use_liger_kernel True \
    --use_flash_attention_2 True \
    --deepspeed './train_args/deepspeed_config/ds_config_zero3.json' \
    --auto_adapt True 2>&1 | tee -a ${logfile}