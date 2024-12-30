
DATA_PATH=''
OUTPUT_PATH=""
MODEL_PATH=""


deepspeed --master_port 29507 --include localhost:0,1 vlm_train.py\
    --train_data_path "$DATA_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --max_seq_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --task_type "QA" \
    --train_mode "lora" \
    --output_dir "$OUTPUT_PATH" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed './train_args/deepspeed_config/ds_config_zero2.json' \
    --bf16 True \
    --torch_dtype bfloat16 \
    --freeze_vision True \
    --freeze_projector False