dataset_name='/workspace/mnt/cmss-wzh/0317_no_sys_data.jsonl'
model_name_or_path='/workspace/alg/c932e29f-31eb-478f-a50f-1262e98adf05/cmss-wangxuejiao/LLaMA-Factory-main/saves/qwen32b/full/Qwen2.5-Coder-32B-Instruct_1223_wad_20250317_reason_sq_8k_gb256_1e-5/checkpoint-200'
save_filename='./0317_no_sys_data/0317_rejected_generate'

nohup python generate.py \
    --dataset_name "$dataset_name" \
    --model_name_or_path "$model_name_or_path" \
    --save_filename "$save_filename" \
    --auto_adapt True \
    --num_completions 3 \
    --temperature 0.8 \
    --response_length 4096 \
    --top_p 0.9 \
    --tensor_parallel_size 8 \
    --chunk_size 50000