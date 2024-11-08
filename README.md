
# LLM-Dojo: 大模型修炼道场 😊
<img src="pic/pic.jpg" width="320">

Tips: 图片完全由AI生成
## 🌟 项目简介
LLM-Dojo使用简洁且易阅读的代码构建模型训练、RLHF框架等各种功能，使项目**易于学习且方便魔改与实验**，与大多开源框架相同均是基于huggingface。
主要内容如下：
- **SFT训练框架:** 简洁清晰的开源大模型训练框架，支持Deepspeed多卡、Lora、QLora、全参等训练，自动适配chat template。
- **RLHF框架:** RLHF训练框架，持续更新，包括 知识蒸馏，DPO、RLOO、SimPO等各种强化学习方法，适配Deepspeed多卡及Lora，一张A100即可运行，详情可见: [RLHF](./rlhf/README.md)。
- **最新LLM tricks详解:** 持续更新大模型领域最新tricks介绍，包括新论文方法的复现等，希望可以给你一些创新的想法，该模块主要集中在```llm_tricks```文件夹下。

### 目录

- [项目简介](#-项目简介)
- [Latest News](#-latest-news)
- [RLHF训练框架](#rlhf训练框架)
  - [已支持RLHF方法](#rlhf训练框架)
  - [知识蒸馏](#rlhf训练框架)
  - [拒绝采样](#rlhf训练框架)
- [SFT训练框架](#sft训练框架)
  - [已支持微调模型](#已支持微调模型)
  - [训练数据格式说明](#训练数据格式说明)
  - [适配框架数据处理](#适配框架数据处理)
  - [Quick Start](#quick-start)
- [Tricks](#tricks)
  - [技术发文](#技术发文)
- [致谢](#-致谢)

## 📖 Latest News
- [2024-11-06] 增加RLHF KTO训练方法
- [2024-11-06] 重构RLHF，具体可见目录中RLHF训练框架部分
- [2024-10-31] 添加auto_adapt参数控制是否自动适配template、更新优化DPO训练(迁移至RLHF目录下)
- [2024-10-15] 增加知识蒸馏训练方法。可见[知识蒸馏](./rlhf/README.md)
- [2024-10-14] 删除chat template模块，因为使用tokenizer的apply_chat_template即可
- [2024-09-20] 增加evaluate模块，一个简洁的模型评测框架，目前仅支持Humaneval。可见[Evaluate](./evaluate/README.md)
- [2024-08-27] 🤓增加从零实现自己编写DPO、SimPO代码，包括数据、loss、训练等部分。可见[DPO example](./llm_tricks/DPO_example/README.md)
- [2024-08-08] 支持直接修改配置文件启动及命令行启动，增加框架适配数据处理代码。
<details> <summary>More news...</summary>

- [2024-08-04] 支持自适应单轮或多轮对话，无需指定单轮或多轮，训练根据数据自行判断单轮或多轮。且可自主设置system命令。可见[训练数据格式说明](#训练数据格式说明)
- [2024-07-19] RLHF 强化学习框架新增CPO,SimPO，以及二者融合CPO-SimPO
- [2024-07-16] RLHF 强化学习框架更新完成，支持deepspeed单卡/多卡 进行强化学习lora、qlora等训练，详细可见[RLHF](./rlhf/README.md)
- [2024-06-9] 🚀支持DPO训练，分为单轮对话DPO(自己构建，方便魔改)和多轮对话DPO(简洁实现)，支持deepspeed的lora和qlora，具体介绍可见 [DPO使用说明](./train_args/dpo/README.md)
- [2024-06-5] 🤓llm_tricks 增加从头开始实现MOE
- [2024-06-10] 🚀增加一步一步实现Transformer技术发文(包括代码等从零介绍)，可见 [技术发文](#技术发文)
- [2024-05-18] 🤓支持Deepspeed单机多卡、单机单卡的Lora、Qlora、全量微调等训练！
- [2024-05-13] 🚀 更新各大模型的Chat Template
- [2024-05-06] 🚀 支持Qwen、Yi模型的Lora、Qlora、Dora微调
- [2024-04-28] 🚀 更新dora微调原理示例、支持qwen模型微调
</details>

## RLHF训练框架

RLHF训练框架，支持并持续更新 知识蒸馏、Reward、PPO、DPO、RLOO、SimPO、KTO等各种强化学习方法，适配Deepspeed多卡及Lora，一张A100即可运行。
详情可见: [RLHF](./rlhf/README.md)。

主要包括三类：

**1、RLHF**

**2、Knowledge Distillation (知识蒸馏)**

**3、Rejected Sampling (拒绝采样) ：待更新**


## SFT训练框架

### 已支持微调模型
理论上支持对所有模型的微调,下述仅为测试过。

支持基于Deepspeed的多卡/单卡 Lora、Qlora、Dora微调:
- [x] [Qwen(Qwen1.5/Qwen2)](https://github.com/QwenLM/Qwen.git)
- [x] [Yi](https://github.com/01-ai/Yi)
- [x] [Gemma系列](https://github.com/google/gemma_pytorch)
- [x] [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- [x] [Deepseek](https://github.com/deepseek-ai/DeepSeek-LLM)
- [x] [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [x] [Llama系列](https://github.com/meta-llama/llama3)
- [x] [deepseek-coder](https://github.com/deepseek-ai/DeepSeek-Coder)
- [x] [哔哩哔哩 Index-1.9B](https://github.com/bilibili/Index-1.9B)
- [x] [baichuan系列](https://github.com/baichuan-inc/Baichuan2)
- [x] [GLM系列](https://github.com/THUDM/GLM-4)

### 😮训练数据格式说明
SFT数据格式为user(system) assistant标准模式,**无需指定单轮或多轮，训练根据数据自行判断单轮或多轮。**

示例如下，示例文件可参见```data/sft_data.jsonl```:
```json lines
{"message": [{"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},{"role": "user", "content": "How many helicopters can a human eat in one sitting"},{"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together"},{"role": "user", "content": "你好"},{"role": "assistant", "content": "hellow"}]}
```
可根据需求自行决定是否增加system字段，**建议训练数据没有特殊需求可删除system字段**


### 适配框架数据处理
鉴于框架指定格式数据可能会跟常规数据有些不同，故可以通过```utils/script/generate_data.py```文件进行处理，输入应为正常的instruction和output的jsonl格式文件，
如下：
```json lines
{"instruction":"将这个句子改写成将来时态：“太阳将会照耀明亮。”","output":"太阳将会散发温暖的光芒。"}
```
运行后即可得到无system的user、assistant指定格式。

### 🤓Quick Start
目前支持直接**python命令单卡训练**、**deepspeed(推荐使用)单机多卡**及**单机单卡训练**. 所有方式均支持Qlora、Lora、Dora方法。


1、 支持**命令行传参**启动，启动示例可见: ```run_example.sh```。 **相关参数在train_args下的common_args.py和sft/base.py。**
```bash
bash run_example.sh
```

2、 也支持**参数文件直接修改默认值**，改好参数后运行以下命令启动：
```bash
deepspeed --include localhost:6,7 main_train.py
```
更详细的Deepspeed原理及解释可以看文章：[Deepspeed配置及使用讲解](https://zhuanlan.zhihu.com/p/698631348)


显存占用测试如下：

| 策略         | 模型大小     | 显存占用 |
|------------|----------|------|
| Lora       | Qwen（7B） | 26g  |
| Lora+Zero2 | Qwen（7B） | 26g  |
| Lora+zero3 | Qwen（7B） | 16g  |

## Tricks
 所有相关的trciks及讲解都在llm_tricks文件夹下
- [Dora代码讲解（llm_tricks/dora/READEME.md）](./llm_tricks/dora/READEME.md)
- [Lora+微调代码实例](https://github.com/mst272/simple-lora-plus)
- [从零实现MOE](./llm_tricks/moe/READEME.md)
- [从零实现DPO](./llm_tricks/DPO_example/README.md)
- [从零实现Transformer](./llm_tricks/transformer/README.md)

### 技术发文
<details> <summary>More news...</summary>

- [Deepspeed配置及使用讲解](https://zhuanlan.zhihu.com/p/698631348)
- [从零代码构建MOE](https://zhuanlan.zhihu.com/p/701777558)
- [一步一步实现Transformer代码](https://medium.com/@sdwzh2725/transformer-code-step-by-step-understandingtransformer-d2ea773f15fa)
- [DPO训练QWEN2及魔改DPO实现](https://zhuanlan.zhihu.com/p/702569978)
- [实现强化学习(RLHF)全流程代码构建（PPO、RLOO等）](https://zhuanlan.zhihu.com/p/708935028)
- [从零实现强化学习DPO（SimPO）训练代码](https://zhuanlan.zhihu.com/p/716706368)
- [实现一个简洁的代码模型评测框架（以Qwen2.5-coder 评测Humaneval为例）](https://zhuanlan.zhihu.com/p/721218072)
- [大模型LLM知识蒸馏代码讲解与训练](https://zhuanlan.zhihu.com/p/1064724364)

</details>


## 🤝 致谢！
项目学习了优秀开源项目，感谢huggingface、流萤等及一些国内外小伙伴的开源项目。

LLM Dojo 期待你的加入。🪂 无论是提出问题（Issue）还是贡献代码（Pull Request），都是对项目的巨大支持。
***
