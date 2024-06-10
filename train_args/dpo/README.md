# 关于DPO训练
目前分为两个模式，分别是multi_dpo和single_dpo。**推荐一般使用multi_dpo**。

DPO训练方式均支持框架中的deepspeed或者python启动模式，相应的lora、qlora也支持。

区别在于两种方式的数据组织形式，前者是使用DPOTrainer自动进行数据处理，且是多轮对话形式，参照格式也可将其改为单轮对话，故前者是单轮与多轮通用的。

后者是自己从零构建的数据组织形式，理论上按照DPOTrainer相同形式，只实现了单轮。这样的**目的是为了更好地理解DPO的过程以及方便一些魔改操作**，权当学习使用。

对于DPO数据，可见```data/dpo_multi_data.jsonl```示例数据。

对于自己构建的single_dpo数据格式，示例为：
```json lines
{"prompt":"哈喽啊","chosen":"你好", "reject": "不好"}
```

## 代码位置

自己构建的single_dpo数据格式代码在```utils/data_process.py```文件中的```DpoDataset```类。

参照官方构建的数据格式在```mian_train.py```中的```load_dpo_dataset```函数里。


## 技术文章
- [DPO训练QWEN2及魔改DPO实现](https://zhuanlan.zhihu.com/p/702569978)


## DPO quick start
### Step1 配置args.py
常规的参数在utils下的args.py，基本默认设置即可，你只需要改一下模型路径、输出路径、task_type、template_name、train_data_path、train_args_path、train_mode等。

使用multi_dpo时args.py中的max_len和max_prompt_length参数是没用的，需要在后面的dpo_config.py中设置

其中:
> train_args_path：为Step2中需要配置的train_args路径

### Step2 配置train_args文件夹下对应文件
相关训练参数在train_args文件夹下对应的文件中。一般就是用```dpo/dpo_config.py```即可

均是采用dataclass格式配置参数，直接在default中修改即可，即不需要直接命令行传输参数了(如果有小伙伴需要这种方式也可以补上)。

在这里修改max_len和max_prompt_length参数，其他需要设置的是是否选择deepspeed模式训练等参数

### Step3 开始训练

开始训练就和之前SFT一样了

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

