
# LLM-Dojo: 大模型修炼道场 😊
<img src="pic/pic.jpg" width="500">

Tips: 图片完全由AI生成
## 🌟 项目简介
欢迎来到 LLM-Dojo，这里是一个开源大模型学习场所(最好的学习永远在项目中)，包括一个每个人都可以以此为基础构建自己的开源大模型训练框架流程、包括各种大模型的tricks实现与原理讲解的llm_tricks模块、主流模型的chat template模版。
主要内容如下：
- ⛳ 1、开源大模型训练框架：每个人都可以根据本项目的代码学习及构建自己的开源大模型训练框架，细节代码主要集中在```utils```文件夹下，训练代码在```main_train.py```，各个部分的构建简洁清晰。
- 🏓2、提供最新LLM tricks的详细讲解及使用：包括最新的微调方法及论文复现等，该模块主要集中在```llm_tricks```文件夹下。
- ⚽ 3、提供主流模型chat template汇总：整合当前主流模型的chat template，以方便自己训练代码时数据处理及微调等操作，该模块主要集中在```chat_template```文件夹下。

"Dojo"一词借用了其在武术训练中的寓意，象征着一个专注于学习和实践的场所。
## 📖 Latest News
- [2024-06-10] 🚀增加一步一步实现Transformer技术发文(包括代码等从零介绍)，可见 [技术发文](#技术发文)
- [2024-06-9] 🚀支持DPO训练，分为单轮对话DPO(自己构建，方便魔改)和多轮对话DPO(简洁实现)，支持deepspeed的lora和qlora，具体介绍可见 [DPO使用说明](./train_args/dpo/README.md)
- [2024-06-5] 🤓llm_tricks 增加从头开始实现MOE
- [2024-05-18] 🤓支持Deepspeed单机多卡、单机单卡的Lora、Qlora等训练！！(即将增加全量训练)
<details> <summary>More news...</summary>

- [2024-05-13] 🚀 更新各大模型的Chat Template
- [2024-05-06] 🚀 支持Qwen、Yi模型的Lora、Qlora、Dora微调
- [2024-04-28] 🚀 更新dora微调原理示例、支持qwen模型微调
</details>

## 🍻 模型 Chat Template总结
 [Chat Template总结](./chat_template/README.md)

在对模型进行微调操作时，数据的输入格式至关重要。

因此，我从官方参考或实现中收集了主流模型的官方模板，都包含在上述文档中，以供大家自己进行微调时参考。

***以下是部分示例：***
### Qwen

官方版本默认的system message即：You are a helpful assistant
```text
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
This is a instruction<|im_end|>
<|im_start|>assistant
This is a answer<|im_end|>
```

### DeepSeek：

官方同样没有提供默认system message，有此需求可依据下述模板自己构建
```text
<｜begin▁of▁sentence｜>This is a system message
User:This is a instruction
Assistant:This is a answer<｜end▁of▁sentence｜>
```

- 无system模式
```text
<｜begin▁of▁sentence｜>User:This is a instruction
Assistant:This is a answer<｜end▁of▁sentence｜>
```

## 📊 项目规划及进展

### 已支持模型
Lora、Qlora、Dora微调:
- [Qwen(Qwen1.5)](https://github.com/QwenLM/Qwen.git)
  - [x] [QWEN]
- [Yi](https://github.com/01-ai/Yi)
  - [x] [Yi]
- [Gemma](https://github.com/google/gemma_pytorch)
  - [x] [Gemma]
- [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
  - [x] [Phi-3]
- [Deepseek](https://github.com/deepseek-ai/DeepSeek-LLM)
  - [x] [Deepseek]

### 已支持tricks及原理讲解
 所有相关的trciks及讲解都在llm_tricks文件夹下
- [Dora代码讲解（llm_tricks/dora/READEME.md）](./llm_tricks/dora/READEME.md)
- [Lora+微调代码实例](https://github.com/mst272/simple-lora-plus)
- [从零实现MOE](./llm_tricks/moe/READEME.md)

### 技术发文
- [Deepspeed配置及使用讲解](https://zhuanlan.zhihu.com/p/698631348)
- [从零代码构建MOE](https://zhuanlan.zhihu.com/p/701777558)
- [一步一步实现Transformer代码](https://medium.com/@sdwzh2725/transformer-code-step-by-step-understandingtransformer-d2ea773f15fa)
- [DPO训练QWEN2及魔改DPO实现](https://zhuanlan.zhihu.com/p/702569978)

## 🤓Quick Start
包括SFT和DPO。

目前支持直接**python命令单卡训练**、**deepspeed单机多卡**及**单机单卡训练**

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

## 😮训练数据
本框架采用的SFT数据格式为***jsonl***形式，```instruction```代表输入，```output```代表输出

示例如下:
```json lines
{"instruction":"将这个句子改写成将来时态：“太阳将会照耀明亮。”","output":"太阳将会散发温暖的光芒。"}
```

对于DPO数据，可见```data/dpo_multi_data.jsonl```示例数据

## 🤝 社区参与
LLM Dojo 期待你的加入！🪂

无论是提出问题（Issue）还是贡献代码（Pull Request），都是对项目的巨大支持。

## 💌 联系方式
- Email: sdwzh272@163.com

***

**感谢！** 📘

项目学习了优秀开源项目，感谢huggingface、流萤等及一些国内小伙伴的开源项目
***
