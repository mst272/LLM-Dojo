DATA_PATH=''
OUTPUT_PATH=""
MODEL_PATH=""

BASE_CMD="CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup accelerate launch --config_file rlhf/ds_config/ds_zero3.yaml main_train.py \
    --train_data_path "$DATA_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --max_len 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --task_type "sft" \
    --train_mode "full" \
    --output_dir "$OUTPUT_PATH" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --warmup_steps 16 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --bf16 True \
    --auto_adapt True"

# 评估相关参数（仅在use_eval_in_train为True时使用）
EVAL_ARGS="--use_eval_in_train True \
    --test_datasets_path "data/test.jsonl" \
    --max_new_tokens 4096 \
    --freq 4 \
    --metrics "code" \
    --vllm_server_port 8001 \
    --vllm_server_timeout 30 \
    --save_best_checkpoints True \
    --max_checkpoints 2 \
    --start_update_best_checkpoints 4 \
    --prompts_apply_chat True \
    --use_vllm True"

# 根据是否需要评估来构建完整命令
if [ "$1" = "--eval" ]; then
    FULL_CMD="$BASE_CMD $EVAL_ARGS"
else
    FULL_CMD="$BASE_CMD --use_eval_in_train False"
fi

# 执行命令
eval $FULL_CMD