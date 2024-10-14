# 使用显卡数量需在yaml文件中修改num_processes参数

# Lora模式， 如需QLora或者全参略微修改参数即可
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --ds_config ./dszero3.yaml ../gkd.py \
    --model_name_or_path deepseek-coder-6.7b-instruct \
    --teacher_model_name_or_path deepseek-coder-33b-instruct\
    --dataset_name ../data_example/gkd_data.jsonl \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gkd-model2 \
    --logging_steps 2 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --lmbda 0.5 \
    --beta 0.5 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --trust_remote_code \
    --bf16 \
    --save_strategy "steps" \
    --save_steps 180 \
    --save_total_limit 5 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --torch_dtype bfloat16 > logs.log 2>&1 &