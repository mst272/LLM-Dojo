# 🥋 LLM-Dojo

> A lightweight playground for `RLHF` and `SFT` experiments, with support for `RLVR`, `KD`, and `Guide-KD`.
>
> 轻量级 RLHF/SFT 实验平台，支持 `RLVR`、`KD` 与 `Guide-KD`。


## 📋 Overview

| 模块 | 说明 |
|------|------|
| [`openrlhf-kd`](./openrlhf-kd/) | 当前主线，基于 OpenRLHF 重构，实现 `RLVR` + `KD` + `Guide-KD` |
| [`main_train.py`](./main_train.py) | 简洁 `SFT` 训练入口 |


## 🎯 RLVR

[`openrlhf-kd`](./openrlhf-kd/) 是这个仓库当前最核心的部分，基于 OpenRLHF 构建，具体训练使用可参见文档 [`openrlhf-kd/examples/README.md`](./openrlhf-kd/examples/README.md)

**主要改动：**

1. 精简框架，只保留 `RLVR` 部分，移除了 `critic` 等不需要的内容
2. 增加 `KD`、`Guide-KD` 与 `reward` 的混合训练，支持按 `datasource` 路由



## ✏️ SFT

根目录的 `SFT` 部分保持了比较简洁的训练入口，适合快速微调实验。

**特性：**

- 支持 `Deepspeed`
- 支持 `LoRA`、`QLoRA`、全参微调
- 自动适配 chat template

示例文件可参见 `data/sft_data.jsonl`

**Quick Start：**

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
