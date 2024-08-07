deepspeed --include localhost:0,1 main_train.py\
    --train_data_path 数据集路径\
    --model_name_or_path  模型路径\
    --task_type sft\
    --train_mode qlora\
    --output_dir 输出路径

# task_type:[pretrain, sft, dpo_multi, dpo_single]

# python main_train.py --train_data_path 数据集路径 --model_name_or_path 模型路径
