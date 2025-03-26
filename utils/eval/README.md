# Train with prediction eval



## Quick start

使用vllm进行生成，其余卡进行训练。

1、启动vllm_serve，例如使用2卡

```shell
CUDA_VISIBLE_DEVICES=0,1 python vllm_serve.py\
    --model "$MODEL_PATH" \
    --tensor_parallel_size 2 \
    --max_model_len 4096
```

2、





## Tip

wandb出问题可以尝试：
pip install wandb==0.12.18