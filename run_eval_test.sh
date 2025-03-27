DATA_PATH=''
OUTPUT_PATH=""
MODEL_PATH=""

# deepspeed 启动命令基础部分（通用参数）
BASE_CMD="deepspeed --master_port 29507 --include localhost:0,1 main_train.py \
    --train_data_path \"$DATA_PATH\" \
    --model_name_or_path \"$MODEL_PATH\" \
    --max_len 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --task_type \"sft\" \
    --train_mode \"qlora\" \
    --output_dir \"$OUTPUT_PATH\" \
    --save_strategy \"steps\" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type \"cosine_with_min_lr\" \
    --gradient_checkpointing True \
    --report_to \"wandb\" \
    --deepspeed './train_args/deepspeed_config/ds_config_zero2.json' \
    --bf16 True \
    --auto_adapt True"

# 评估相关参数（仅在use_eval_in_train为True时使用）
EVAL_ARGS="--use_eval_in_train True \
    --test_datasets_path \"./eval_train_test/test.jsonl\" \
    --eval_num_samples 6 \
    --eval_freq 1 \
    --eval_max_checkpoints 1 \
    --eval_batch_size 1 \
    --eval_start_steps 100"

# 根据是否需要评估来构建完整命令
if [ "$1" = "--eval" ]; then
    FULL_CMD="$BASE_CMD $EVAL_ARGS"
else
    FULL_CMD="$BASE_CMD --use_eval_in_train False"
fi

# 执行命令
eval $FULL_CMD