MODELS_PATH="/qwen"
LOGS_PATH="./logs.jsonl"
OUT_PATH='./out.jsonl'
METRIC_PATH='./metric.json'
DATA_FILE='./dataset/humaneval_python.jsonl'


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name_or_path "$MODELS_PATH" \
    --task_name "humaneval" \
    --save_logs_path "$LOGS_PATH" \
    --output_path "$OUT_PATH" \
    --do_sample false \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --evaluate_only false \
    --torch_dtype "bf16" \
    --save_metrics_path $METRIC_PATH \
    --data_file $DATA_FILE