# LLM-Dojo

> A lightweight playground for `RLHF` and `SFT` experiments.

当前仓库主要有两条线：

- [`openrlhf-kd`](./openrlhf-kd/): 当前主线，基于 `OpenRLHF` 重构，实现 `RLVR + KD + Guide-KD`
- [`main_train.py`](./main_train.py): 简洁 `SFT` 训练入口


## RLHF

[`openrlhf-kd`](./openrlhf-kd/) 是这个仓库当前最核心的部分，基于openrlhf构建，具体训练使用可参见文档 [`openrlhf-kd/examples/README.md`](./openrlhf-kd/examples/README.md)

主要改动：

1. 精简框架，只保留 `RLVR` 部分，移除了 `critic` 等不需要的内容。
2. 增加 `KD`、`Guide-KD` 与 `reward` 的混合训练，支持按 `datasource` 路由。

## SFT

根目录的 `SFT` 部分保持了比较简洁的训练入口，适合快速微调实验。

- 支持 `Deepspeed`
- 支持 `LoRA`、`QLoRA`、全参微调
- 自动适配 chat template

示例文件可参见```data/sft_data.jsonl```

Quick start：

```bash
bash run_example.sh
```

或：

```bash
deepspeed --include localhost:0,1 main_train.py \
  --train_data_path /path/to/data.jsonl \
  --model_name_or_path /path/to/model \
  --task_type sft \
  --train_mode qlora \
  --output_dir /path/to/output
```