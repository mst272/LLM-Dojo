# LLM-Dojo

轻量、可读、便于实验的大模型训练仓库，当前主要包含两条主线：

- `LLM-Dojo` 根目录下的简洁 `SFT` 框架
- 基于 `OpenRLHF` 改造的 `openrlhf-kd`，聚焦 `RLVR + KD + Guide-KD`

旧版说明已保留在 [README_legacy.md](./README_legacy.md)。

## 项目概览

| 方向 | 位置 | 说明 |
| --- | --- | --- |
| SFT | `main_train.py` | 面向日常监督微调，强调代码简洁和可改性 |
| RLVR / KD | `openrlhf-kd/` | 基于 OpenRLHF 精简与重构，聚焦 reward 驱动训练和蒸馏混合 |
| Tricks / Notes | `llm_tricks/` | 一些训练细节、实现拆解和实验记录 |

## 1. LLM-Dojo SFT

根目录的 `SFT` 部分是一个偏工程实用、同时尽量保持清晰的训练入口，适合快速开始微调实验，也适合继续向数据处理、模板适配和训练策略上扩展。

### 特性

- 支持 `Deepspeed` 单机单卡 / 多卡训练
- 支持 `LoRA`、`QLoRA`、全参微调
- 自动适配 chat template
- 训练入口简单，适合学习和二次开发

### 数据格式

训练数据采用标准对话格式，核心字段为 `message`：

```json
{
  "message": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，请问有什么可以帮你？"}
  ]
}
```

示例数据可参考 [`data/`](./data/)。

### Quick Start

最简单的方式是直接参考 [`run_example.sh`](./run_example.sh)：

```bash
bash run_example.sh
```

也可以直接使用 `deepspeed` 启动：

```bash
deepspeed --include localhost:0,1 main_train.py \
  --train_data_path /path/to/data.jsonl \
  --model_name_or_path /path/to/model \
  --task_type sft \
  --train_mode qlora \
  --output_dir /path/to/output
```

训练配置可继续在 [`train_args/`](./train_args/) 中扩展。

## 2. openrlhf-kd

[`openrlhf-kd/`](./openrlhf-kd/) 基于 `OpenRLHF` 构建和整理的一条更聚焦的训练分支，目标不是保留原框架的全部能力，而是把我当前真正使用的一条研究路径做得更清晰、更可控。

### 相比原始 OpenRLHF 的主要改动

1. 精简框架，只保留 `RLVR` 主链路，移除了 `critic` 以及当前实验不需要的其他部分，整体代码更短、更直观，也更方便继续改。
2. 在 reward 训练路径上增加了 `KD`、`Guide-KD` 与 `reward` 的混合机制，支持按 `datasource` 做路由与组合。

### 当前支持的训练形态

- 纯 `RLVR`
- 纯 `KD`
- 纯 `Guide-KD`
- `RLVR + KD` 混合
- 按 `datasource` 的部分混合与多教师路由

### 文档入口

更完整的配置说明、模式说明和样例请看：

- [`openrlhf-kd/examples/README.md`](./openrlhf-kd/examples/README.md)
- [`openrlhf-kd/examples/guide-kd-reward.py`](./openrlhf-kd/examples/guide-kd-reward.py)

### Minimal Example

```bash
python -m openrlhf.cli.train_ppo_ray \
  --pretrain /path/to/model \
  --prompt_data /path/to/dataset \
  --input_key input \
  --label_key label \
  --remote_rm_url examples/guide-kd-reward.py \
  --advantage_estimator reinforce
```

如果需要启用混合蒸馏，可以在此基础上增加：

```bash
--kd_coef 0.3
```

## 适合什么场景

- 想快速跑一个可读、可改的 `SFT` 微调框架
- 想在 `reward / RLVR / KD / Guide-KD` 之间做组合实验
- 想基于相对简洁的代码继续扩展自己的训练链路

## 说明

这个仓库更偏“研究与实现并行”的风格：一方面保持入口简洁，另一方面保留足够多的改造空间，方便围绕训练流程本身继续做实验。
