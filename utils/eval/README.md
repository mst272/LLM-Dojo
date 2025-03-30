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

可能出现的问题：
1、直接deepspeed --master_port 29508 --include localhost:2,3,4,5,6,7 main_train.py保存checkpoint时有问题，所以建议
accelerate launch --config_file rlhf/ds_config/ds_zero3.yaml main_train.py

2、设置zero3_init_flag: false，保存模型才没有问题，不然可能出现cpu oom.尚不知原因。

3、上述问题貌似破案了，eval代码时出现了内存泄漏。