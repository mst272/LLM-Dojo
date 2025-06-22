#!/bin/bash

# 在租赁的训练平台中使用，进入指定环境
. /opt/conda/etc/profile.d/conda.sh && conda activate your_environment
conda info -e

export SWANLAB_API_KEY='xx'


export FORCE_TORCHRUN=1
export NNODES=$PET_NNODES
export NODE_RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export PET_NPROC_PER_NODE=$PET_NPROC_PER_NODE

export DISABLE_VERSION_CHECK=1
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE}"



# NCCL 网络优化
export NCCL_SOCKET_IFNAME=eth0     # 指向 IB/40G网口
export NCCL_IB_DISABLE=0            # 使用 RDMA
export NCCL_P2P_DISABLE=0



# # Debug 打印（可选，用于排查瓶颈）
# export NCCL_DEBUG=INFO                  # 或 WARN；默认关闭可设空
# export NCCL_ASYNC_ERROR_HANDLING=1      # 遇错不死锁




DATA_PATH=''
OUTPUT_PATH=""

MODEL_PATH=""
logfile=''

# task_type:[sft]  pretrain正在开发
# train_mode:[qlora, lora, full]
# train_args_path: [sft_args,dpo_args]
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=${PET_NPROC_PER_NODE} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main_train.py \
    --train_data_path "$DATA_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --max_len 8192 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --task_type "sft" \
    --train_mode "full" \
    --output_dir "$OUTPUT_PATH" \
    --save_strategy "steps" \
    --save_steps 650 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "swanlab" \
    --fp16 True \
    --bf16 False \
    --deepspeed './train_args/deepspeed_config/ds_config_zero3.json' \
    --auto_adapt True 2>&1 | tee -a ${logfile}