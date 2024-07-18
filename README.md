
# LLM-Dojo: 大模型修炼道场 😊
<img src="pic/pic.jpg" width="320">

Tips: 图片完全由AI生成
## 🌟 项目简介
不同于其他优秀的开源训练框架的高度封装与集成，LLM-Dojo使用简洁且易阅读的代码构建模型训练、RLHF框架等各种功能，使项目**易于学习**，每个人都能以此项目为基础自己构建与理解，且与大多开源框架相同均是基于huggingface，性能并不会有太多出入。
主要内容如下：
- **开源大模型训练框架:** 简洁清晰的开源大模型训练框架，支持Deepspeed多卡、Lora(Dora)、QLora、全参等训练，细节代码主要集中在```utils```文件夹下，训练代码在```main_train.py```。
- **RLHF框架:** RLHF训练框架，支持并持续更新Reward训练、PPO、DPO、RLOO、SimPO等各种强化学习方法，适配Deepspeed多卡及Lora，一张A100即可运行，详情可见: [RLHF](./rlhf/README.md)。
- **最新LLM tricks详解:** 持续更新大模型领域最新tricks介绍，包括新论文方法的复现等，希望可以给你一些创新的想法，该模块主要集中在```llm_tricks```文件夹下。
- **主流模型chat template汇总:** 整合当前主流模型的chat template，以方便自己训练代码时数据处理及微调等操作，详情可见: [Chat Template](./chat_template/README.md)。

### 目录

- [项目简介](#-项目简介)
- [Latest News](#-latest-news)
- [RLHF训练框架](#rlhf训练框架)
- [项目规划及进展](#-项目规划及进展)
  - [已支持微调模型](#已支持微调模型)
  - [已更新tricks讲解](#已更新tricks讲解)
  - [Chat Template汇总](#chat-template汇总)
  - [技术发文](#技术发文)
- [训练数据格式说明](#训练数据格式说明)
- [Quick Start](#quick-start)
- [致谢](#-致谢)

## 📖 Latest News
- [2024-07-19] 🤓RLHF 强化学习框架新增CPO,SimPO，以及二者融合CPO-SimPO
- [2024-07-16] 🤓RLHF 强化学习框架更新完成，支持deepspeed单卡/多卡 进行强化学习lora、qlora等训练，详细可见[RLHF](./rlhf/README.md)
- [2024-06-10] 🚀增加一步一步实现Transformer技术发文(包括代码等从零介绍)，可见 [技术发文](#技术发文)
- [2024-06-9] 🚀支持DPO训练，分为单轮对话DPO(自己构建，方便魔改)和多轮对话DPO(简洁实现)，支持deepspeed的lora和qlora，具体介绍可见 [DPO使用说明](./train_args/dpo/README.md)
- [2024-06-5] 🤓llm_tricks 增加从头开始实现MOE
- [2024-05-18] 🤓支持Deepspeed单机多卡、单机单卡的Lora、Qlora、全量微调等训练！！(即将增加全量训练)
<details> <summary>More news...</summary>

- [2024-05-13] 🚀 更新各大模型的Chat Template
- [2024-05-06] 🚀 支持Qwen、Yi模型的Lora、Qlora、Dora微调
- [2024-04-28] 🚀 更新dora微调原理示例、支持qwen模型微调
</details>

## RLHF训练框架

RLHF训练框架，支持并持续更新Reward训练、PPO、DPO、RLOO、SimPO等各种强化学习方法，适配Deepspeed多卡及Lora，一张A100即可运行。
详情可见: [RLHF](./rlhf/README.md)。

## 📊 项目规划及进展

### 已支持微调模型
支持基于Deepspeed的多卡/单卡 Lora、Qlora、Dora微调:
- [x] [Qwen(Qwen1.5/Qwen2)](https://github.com/QwenLM/Qwen.git)
- [x] [Yi](https://github.com/01-ai/Yi)
- [x] [Gemma](https://github.com/google/gemma_pytorch)
- [x] [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- [x] [Deepseek](https://github.com/deepseek-ai/DeepSeek-LLM)
- [x] [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [x] [Llama系列](https://github.com/meta-llama/llama3)
- [x] [deepseek-coder](https://github.com/deepseek-ai/DeepSeek-Coder)
- 待更新GLM、baichuan
### 已更新tricks讲解
 所有相关的trciks及讲解都在llm_tricks文件夹下
- [Dora代码讲解（llm_tricks/dora/READEME.md）](./llm_tricks/dora/READEME.md)
- [Lora+微调代码实例](https://github.com/mst272/simple-lora-plus)
- [从零实现MOE](./llm_tricks/moe/READEME.md)

### Chat Template汇总
-  [Chat Template总结](./chat_template/README.md)

### 技术发文
- [Deepspeed配置及使用讲解](https://zhuanlan.zhihu.com/p/698631348)
- [从零代码构建MOE](https://zhuanlan.zhihu.com/p/701777558)
- [一步一步实现Transformer代码](https://medium.com/@sdwzh2725/transformer-code-step-by-step-understandingtransformer-d2ea773f15fa)
- [DPO训练QWEN2及魔改DPO实现](https://zhuanlan.zhihu.com/p/702569978)

## 😮训练数据格式说明
本框架采用的SFT数据格式为***jsonl***形式，```instruction```代表输入，```output```代表输出

示例如下:
```json lines
{"instruction":"将这个句子改写成将来时态：“太阳将会照耀明亮。”","output":"太阳将会散发温暖的光芒。"}
```

对于DPO数据，可见```data/dpo_multi_data.jsonl```示例数据


## 🤓Quick Start
包括SFT和DPO。

目前支持直接**python命令单卡训练**、**deepspeed单机多卡**及**单机单卡训练**。

所有方式均支持Qlora、Lora、Dora方法。

### SFT微调(FineTune)

#### Step1 配置args.py
不同的微调方法有不同的配置，但大体都是类似的，基本默认设置即可，你只需要改一下模型路径、输出路径等等。

常规的参数在utils下的args.py。

其中:
> train_args_path：为Step2中需要配置的train_args路径

#### Step2 配置train_args文件夹下对应文件
相关训练参数在train_args文件夹下对应的文件中。一般就是用```base.py```即可
均是采用dataclass格式配置参数，直接在default中修改即可，即不需要直接命令行传输参数了(如果有小伙伴需要这种方式也可以补上)。

#### Step3 开始训练

😶Python命令单卡启动：

设置好相关配置后即可运行main_train.py进行训练
```bash
python main_train.py
```

🙃Deepspeed单卡或多卡启动：

使用Deepspeed训练时前两步与常规相同，但需要额外配置ds_config文件，项目中已给出常用的配置示例，位于```train_args/deepspeed_config/```路径下，
更详细的Deepspeed原理及解释可以看文章：[Deepspeed配置及使用讲解](https://zhuanlan.zhihu.com/p/698631348)

运行以下命令启动：
```bash
deepspeed --include localhost:6,7 main_train.py
```
其中```include localhost```参数用于选择训练的GPU，可选单卡也可选多卡。

显存占用测试如下：

| 策略         | 模型大小     | 显存占用 |
|------------|----------|------|
| Lora       | Qwen（7B） | 26g  |
| Lora+Zero2 | Qwen（7B） | 26g  |
| Lora+zero3 | Qwen（7B） | 16g  |

### DPO
目前区分single_dpo和multi_dpo模式，前者是自己实现dataset并映射，以供大家魔改使用。 
后者采用官方示例，故建议使用后者。具体使用说明可见：[DPO使用说明](./train_args/dpo/README.md)

### 推理(Infer)

## 🤝 致谢！
项目学习了优秀开源项目，感谢huggingface、流萤等及一些国内外小伙伴的开源项目。

LLM Dojo 期待你的加入。🪂 无论是提出问题（Issue）还是贡献代码（Pull Request），都是对项目的巨大支持。
***
