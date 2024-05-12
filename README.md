
# LLM-Dojo: 大模型修炼道场 😊
<img src="pic/pic.jpg" width="500">

Tips: 图片完全由AI生成
## 🌟 项目简介
欢迎来到 LLM-Dojo，这里是一个开源大模型学习场所(最好的学习永远在项目中)，包括一个开源大模型训练框架，以及llm_tricks模块，其中包括各种大模型的tricks实现与原理讲解！
主要内容如下：
- ⛳ 开源大模型训练框架，代码主要集中在```utils```
- ⛳ 提供最新LLM tricks的详细讲解及使用
- ⛳ 整合当前主流模型的chat template 以方便微调查阅及使用(待更新)

"Dojo"一词借用了其在武术训练中的寓意，象征着一个专注于学习和实践的场所。
## 📖 Latest News
- [2024-05-13] 🚀 更新各大模型的Chat Template
<details> <summary>More news...</summary>

- [2024-05-06] 🚀 支持Qwen、Yi模型的Lora、Qlora、Dora微调
- [2024-04-28] 🚀 更新dora微调原理示例、支持qwen模型微调
</details>

## 🍻 模型 Chat Template总结
 [Chat Template总结](./chat_template/README.md)

## 📊 项目规划及进展

### 已支持模型
- [Qwen](https://github.com/QwenLM/Qwen.git)
  - [x] [QWEN Lora、Qlora、Dora微调]
- [Yi](https://github.com/01-ai/Yi)
  - [x] [Yi Lora、Qlora、Dora微调]
- [Gemma](https://github.com/google/gemma_pytorch)
  - [x] [Gemma Lora、Qlora、Dora微调]
- [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
  - [x] [Phi-3 Lora、Qlora、Dora微调]

### 已支持tricks及原理讲解
 所有相关的trciks及讲解都在llm_tricks文件夹下
- [Dora代码讲解（llm_tricks/dora/READEME.md）](./llm_tricks/dora/READEME.md)

## 🤓Quick Start
项目还在初始阶段， 目前仅支持单卡训练。建议使用Qlora。

### 微调训练(FineTune)

#### Step1 配置args.py
不同的微调方法有不同的配置，但大体都是类似的。常规的参数在utils下的args.py。

其中:
> train_args_path：为Step2中需要配置的train_args路径

#### Step2 配置train_args文件夹下对应文件
相关训练参数在train_args文件夹下对应的模型中。
均是采用dataclass格式配置参数，直接在default中修改即可，即不需要直接命令行传输参数了(如果有小伙伴需要这种方式也可以补上)。

#### Step3 开始训练
设置好相关配置后即可运行main_train.py进行训练
```sh
python main_train.py
```
### 推理(Infer)

## 😮训练数据
待更新、、、、

## 🤝 社区参与
LLM Dojo 期待你的加入！🪂

无论是提出问题（Issue）还是贡献代码（Pull Request），都是对项目的巨大支持。

## 💌 联系方式
- GitHub: LLM-Dojo
- Gitter: mst272
- Email: sdwzh272@163.com

***

**感谢！** 📘

项目学习了优秀开源项目，感谢huggingface及一些国内小伙伴的开源项目
***
