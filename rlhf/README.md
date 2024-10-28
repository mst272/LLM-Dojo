# RLHF 强化学习框架

不同于其他框架实现的高度封装的强化学习框架，本框架使用简洁的代码基于TRL对各种强化学习方法进行了集成，便于自己修改与学习，是一个轻量化的强化学习框架。

主要资源是在1-8张40G A100上进行实验，支持lora qlora 及deepspeed单卡或多卡训练。 一些细节问题还需要后续的优化。

主要包括三类：

**1、RLHF**

**2、Knowledge Distillation (知识蒸馏)**

**3、Rejected Sampling (拒绝采样) ：待更新**

## 目录

- [RLHF](#rlhf)
  - [目前支持的强化学习方法](#目前支持的rlhf)
  - [Quick Star](#quick-star)
    - [数据格式要求](#数据格式要求)
    - [Step1 训练Reward Model](#step1-训练reward-model)
    - [Step2 选择rlhf方法如dpo等](#step2-选择rlhf方法如dpo等)
    - [注意事项](#注意事项)
  - [显存实验](#显存实验)
- [Knowledge Distillation](#knowledge-distillation)
  - [Quick Star](#quick-star-1)
- [感谢](#感谢)

## RLHF
### 目前支持的RLHF
支持单轮形式和多轮形式。不过实践来看主要的训练方式即为单轮。

- ✅ Reward模型的训练
- ✅ RLOO
- ✅ PPO

不需要训练reward：
- ✅ SimPO
- ✅ CPO
- ✅ CPO-SimPO
- ✅ DPO：详情见 [DPO](../train_args/dpo/README.md)，历史原因还没来得及合并到此目录下。

### Quick Star

对于PPO和RLOO，需要训练reward模型。

对于其余方法，则不需要训练reward模型。

#### 数据格式要求

数据格式一般要求有如下三个字段:
- prompt
- chosen
- rejected

huggingface上也有很多数据集，例如：```trl-internal-testing/hh-rlhf-helpful-base-trl-style```，但其chosen和rejected中包含了prompt，并不严格符合数据格式。

因为我们要构建模型的的chat template，故数据格式稍有不同，prompt中必须包含```role```和```content```字段。且在chosen和rejected中分离了prompt。

本框架采用的数据格式为jsonl，具体可见示例数据：```rlhf/data_example/data.jsonl```。三个字段prompt、chosen 、rejected彼此分离，训练时再进行组合。

此数据集可用于上述所有训练使用。且示例数据只是单轮，如若需要构建多轮，可以将多轮对话写在prompt中，最终的assistant回答分别写在chosen和rejected中(针对多轮其实还可以取每一轮的user当prompt，后面有时间可能会实现)。


#### Step1 训练Reward Model

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

#### Step2 选择RLHF方法，如DPO等

**配置相关参数**

1、需要配置两个参数文件，第一个为```common_args.py```,主要是配置训练方式(Lora/Qlora)及RLHF优化方法(PPO、RLOO等)等。

2、第二个文件为RLHF优化方法的相关文件, 主要都在```rlhf_args```文件夹内

**deepspeed启动**

注：使用deepspeed时需要通过accelerate进行使用，直接deepspeed的话会报错(目前似乎没有很好的解决方案)

```bash
CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --config_file ./ds_config/deepspeed_zero3.yaml train_rlhf.py
```
运行上述命令，参数解释如下：
- CUDA_VISIBLE_DEVICES：代表你要用的卡，可以指定多块，但是要在deepspeed_zero3.yaml文件中修改```num_processes```为对应数量
- config_file: deepspeed的yaml文件路径，在```ds_config```文件夹下

#### 注意事项
1、需要自己去看AutoModelForSequenceClassification是否可以加载其Classification模型，不能的话需要在其config文件中映射。

2、涉及到reward模型时，需要两个模型的tokenizer相同。

3、一般来说trl的trainer是不支持使用deepspeed的optimizer和scheduler的

4、不支持Qlora和deepspeed zero-3，支持Qlora和deepspeed zero-2


#### 显存实验
res_length为64

| **RLHF** | **deepspeed** | **方式** | **Reward Model** | **SFT Model**  | **显存占用**               |
|----------|---------------|--------|------------------|----------------|------------------------|
| RLOO     | Zero 3        | Lora   | QWEN2(7B)        | QWEN2(7B)      | 2 x A100(40GB): 15~30G |
| RLOO     | Zero 3        | Full   | QWEN2(7B)        | QWEN2(7B)      | 2 x A100(40GB): 速度很慢   |
| RLOO     | Zero 2        | Qlora  | QWEN2(7B)        | QWEN2(7B)      | 2 x A100(40GB): 30~40G |
| PPO      | Zero 2        | Lora   | MiniCPM(2B)      | Deepseek(6.7B) | 2 x A100(40GB): OOM    |
| PPO      | Zero 3        | Lora   | MiniCPM(2B)      | Deepseek(6.7B) | 2 x A100(40GB): 20-25G |
| PPO      | Zero 2        | Qlora  | MiniCPM(2B)      | Deepseek(6.7B) | 2 x A100(40GB): 30G    |

## Knowledge Distillation
目前支持三种类型的知识蒸馏，GKD效果最好：
- Supervised KD(off-policy)
- SeqKD(off-policy)
- GKD(on-policy)

具体介绍可参见文章：

### Quick Star
进入script目录下bash运行```gkd_run.sh```即可，修改对应参数运行。同样支持Deepspeed，参数介绍可看上述文章。

**参数介绍**：
- lmbda：0时为Supervised KD，1时为GKD。可在[0,1]范围内选择，这样就会混合比例
- beta:  0时loss为KLD， 1时为JSD。可在[0,1]范围内选择，这样就会混合比例
- seq_kd: True时Supervised KD将替换为Seq KD，默认为False，其他不变。（最近才合并的PR，trl还没有更新，暂时先写下）
- model_name_or_path：Student Model，即你需要训练的模型
- teacher_model_name_or_path：Teacher Model, 不训练。

## Rejected Sampling
待更新

## 感谢

特别感谢huggingface trl做出的强大贡献，通过 trl 我们真的可以很容易简洁的实现RLHF。