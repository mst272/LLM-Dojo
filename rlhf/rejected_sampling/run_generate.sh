dataset_name=''
model_name_or_path=''
save_filename='rejected_generate'

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