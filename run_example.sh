
DATA_PATH=''
OUTPUT_PATH=""
MODEL_PATH=""

# task_type:[pretrain, sft, dpo_multi, dpo_single]
# train_mode:[qlora, lora, full]
# train_args_path: [sft_args,dpo_args]

# deepspeed 启动
deepspeed --include localhost:0,1 main_train.py\
    --train_args_path "sft_args" \
    --train_data_path "$DATA_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --max_len 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --task_type "sft" \
    --train_mode "qlora" \
    --output_dir "$OUTPUT_PATH" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed './train_args/deepspeed_config/ds_config_zero2.json' \
    --bf16 True





# python main_train.py --train_data_path 数据集路径 --model_name_or_path 模型路径 ......同上述传入参数
