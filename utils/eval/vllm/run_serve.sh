MODEL_PATH='Qwen2.5-Coder-32B-Instruct'


CUDA_VISIBLE_DEVICES=0,1 python vllm_serve.py\
    --model "$MODEL_PATH" \
    --tensor_parallel_size 2 \
    --max_model_len 4096 \
    --port 8001 \
    --dtype "bfloat16" \
    --enable_prefix_caching False