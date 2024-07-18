# RLHF 强化学习框架

不同于其他框架实现的高度封装的强化学习框架，本框架使用简洁的代码对各种强化学习方法进行了集成，且便于自己修改与学习，是一个轻量化的强化学习框架。

主要资源是在1-8张40G A100上进行实验，支持lora qlora 及deepspeed单卡或多卡训练。 一些细节问题可能需要后续的优化,有想法伙伴可以提个PR一起优化这个项目。

## 目录

- [目前支持的强化学习方法](#目前支持的强化学习方法)
- [Quick Star](#quick-star)
  - [数据格式要求](#数据格式要求)
  - [Step1 训练Reward Model](#step1-训练reward-model)
  - [Step2 基于不同优化方法进行强化学习，如PPO等](#step2-基于不同优化方法进行强化学习如ppo等)
  - [注意事项](#注意事项)
  - [参数解释](#参数解释)
- [支持矩阵](#支持矩阵)
- [显存实验](#显存实验)
- [感谢](#感谢)


## 目前支持的强化学习方法
支持RLHF的Lora、Dora、Qlora、全量参数训练。

- ✅ Reward模型的训练
- ✅ RLOO
- ✅ PPO
- ✅ SimPO
- ✅ CPO
- ✅ CPO-SimPO



## Quick Star

### 数据格式要求

数据格式要求有如下三个字段:
- prompt
- chosen
- rejected

reward阶段需要chosen和rejected， RL阶段只需要prompt字段。

huggingface上也有很多数据集，例如：```trl-internal-testing/hh-rlhf-helpful-base-trl-style```，因为我们要构建模型的的chat template，故数据格式稍有不同，prompt中必须包含```role```和```content```字段。

数据格式为jsonl，具体可见示例数据：```rlhf/data_example/data.jsonl```


### Step1 训练Reward Model

**配置相关参数**

1、需要配置两个参数文件，都在```reward_args```内，第一个为```model_config.py```,主要配置模型相关，如是否lora、qlora等。

2、第二个在```model_config.py```，主要配置训练相关参数。

**启动**

显存占用不算高，可以直接命令启动，也可以deepspeed启动(具体可见Step2中介绍)。
```bash
CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --config_file ./ds_config/deepspeed_zero3.yaml reward_model.py
```

```bash
python reward_model.py
```

注：
训练Qwen2时遇到报错，提示```no padding token is defined```。需要在qwen2 ```config.json```中添加pad_token_id,在tokenizer中设置没用。

### Step2 基于不同优化方法进行强化学习，如PPO等

**配置相关参数**

1、需要配置两个参数文件，第一个为```common_args.py```,主要是配置训练方式(Lora/Qlora)及RLHF优化方法(PPO、RLOO等)等。

2、第二个文件为RLHF优化方法的相关文件, 主要都在```rlhf_args```文件夹内

**deepspeed启动**

注：使用deepspeed时需要通过accelerate进行使用，直接deepspeed的话会报错(目前似乎没有很好的解决方案)

```bash
CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --config_file ./ds_config/deepspeed_zero3.yaml rlhf_train.py
```
运行上述命令，参数解释如下：
- CUDA_VISIBLE_DEVICES：代表你要用的卡，可以指定多块，但是要在deepspeed_zero3.yaml文件中修改```num_processes```为对应数量
- config_file: deepspeed的yaml文件路径，在```ds_config```文件夹下

### 注意事项
1、需要自己去看AutoModelForSequenceClassification是否可以加载其Classification模型，不能的话需要在其config文件中映射。

2、涉及到reward模型时，需要两个模型的tokenizer相同。

3、一般来说trl的trainer是不支持使用deepspeed的optimizer和scheduler的

4、不支持Qlora和deepspeed zero-3，支持Qlora和deepspeed zero-2



### 参数解释

The num_train_epochs and num_ppo_epochs are actually two different things. The num_train_epochs means how many epochs do we go over the dataset, the num_ppo_epochs means the number of epochs we perform PPO updates on a batch of data. So, there is a subtle but meaningful difference here.



## 支持矩阵
✅ 代表支持deepspeed 全策略

| 支持方法/deepspeed | LORA(Dora) | QLORA | Full | Unsloth(待更新) |
|----------------|------------|-------|------|--------------|
| RLOO           | ✅          | Zero2 | ✅    | ❌            |
| PPO            | ✅          | Zero2 | ✅    | ❌            |
| SimPO          |            |       |      | ❌            |




## 显存实验
res_length为64

| **RLHF** | **deepspeed** | **方式** | **Reward Model** | **SFT Model**  | **显存占用**               |
|----------|---------------|--------|------------------|----------------|------------------------|
| RLOO     | Zero 3        | Lora   | QWEN2(7B)        | QWEN2(7B)      | 2 x A100(40GB): 15~30G |
| RLOO     | Zero 3        | Full   | QWEN2(7B)        | QWEN2(7B)      | 2 x A100(40GB): 速度很慢   |
| RLOO     | Zero 2        | Qlora  | QWEN2(7B)        | QWEN2(7B)      | 2 x A100(40GB): 30~40G |
| PPO      | Zero 2        | Lora   | MiniCPM(2B)      | Deepseek(6.7B) | 2 x A100(40GB): OOM    |
| PPO      | Zero 3        | Lora   | MiniCPM(2B)      | Deepseek(6.7B) | 2 x A100(40GB): 20-25G |
| PPO      | Zero 2        | Qlora  | MiniCPM(2B)      | Deepseek(6.7B) | 2 x A100(40GB): 30G    |




## 感谢

特别感谢huggingface trl做出的强大贡献，通过 trl 我们真的可以很容易简洁的实现RLHF。