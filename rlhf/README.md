# RLHF 强化学习框架

本框架使用简洁的代码基于Huggingface对各种强化学习方法进行了集成，便于自己修改与使用，是一个轻量化的强化学习框架。

主要资源是在1-8张40G A100上进行实验，支持lora qlora 及deepspeed单卡或多卡训练。

主要包括三类：

**1、RLHF**

**2、Knowledge Distillation (知识蒸馏)**

**3、Rejected Sampling (拒绝采样) ：待更新**

## 目录

- [RLHF](#rlhf)
  - [目前支持的RLHF](#目前支持的rlhf)
  - [Quick Star](#quick-star)
    - [数据格式要求](#数据格式要求)
    - [数据格式选择](#数据格式选择)
    - [启动训练](#启动训练)
    - [注意事项](#注意事项)
  - [显存实验](#显存实验)
- [Knowledge Distillation](#knowledge-distillation)
  - [Quick Star](#quick-star-1)
- [感谢](#感谢)

## RLHF
### 目前支持的RLHF
实践来看主要的训练方式即为单轮。

- ✅ Reward模型的训练
- ✅ RLOO
- ✅ PPO(暂时不可用)
- ✅ SimPO
- ✅ CPO
- ✅ CPO-SimPO
- ✅ DPO
- ✅ KTO

### 🚀Quick Star

若有问题请尝试 deepspeed==0.15.4/python==3.10, 或者出现loss、rewards/chosen为nan时，请查看当前目录下的requirements.txt，按照此版本安装看是否能解决。

一些潜在的问题，暂时还没得到解决或者潜在的解决方案：

https://github.com/huggingface/alignment-handbook/issues/57

https://github.com/microsoft/DeepSpeed/issues/6793#issuecomment-2502620884

https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/issues/29

#### 数据格式要求
✅ DPO、CPO、SimPO、CPO-SimPO:

需要有如下字段：
- prompt
- chosen
- rejected

```json lines
{"prompt":[{"role":"user","content":"How are you?"}],"chosen":[{"role":"assistant","content":"fine"}],"rejected":[{"role":"assistant","content":"no"}]}
```
✅ KTO:
- prompt
- completion
- label

比较特殊,相当于chosen的label为true,rejected的label为false：
```json lines
{"prompt":[{"role":"user","content":"How are you?"}],"completion":[{"role":"assistant","content":"fine"}],"label":true}
```

✅ Reward:
- chosen
- rejected

```json lines
{"chosen":[{"role":"user","content":"How are you?"},{"role":"assistant","content":"fine"}],"rejected":[{"role":"user","content":"How are you?"},{"role":"assistant","content":"no"}]}
```
✅ DPO、RLOO:
- prompt

```json lines
{"prompt":[{"role":"user","content":"How are you?"}]}
```

#### 数据格式选择

**1.自动适配Chat Template格式**: 输入数据需为user assistant标准模式,具体可见上述数据格式要求。

**2.不使用Chat格式**: 输入数据直接改为相应字段格式即可，例如:
```json lines
{"prompt":"How are you?","chosen":"fine", "rejected": "no"}
```

```json lines
{"chosen":"How are you? fine", "rejected": "How are you? no"}
```
训练时便不会进行适配，采用原始输入进行训练。


#### 启动训练

两个参数配置文件，第一个为```common_args.py```, 其余不同方法的配置在```rlhf_args```文件夹内

建议使用deepspeed启动，启动脚本在```rlhf_run.sh```
```bash
bash rlhf_run.sh
```

 - rlhf_type: [PPO,RLOO,CPO,DPO,SimPO,CPOSimPO,Reward]
 - train_mode: [lora, qlora, full]

#### 注意事项
1、需要自己去看AutoModelForSequenceClassification是否可以加载其Classification模型，不能的话需要在其config文件中映射。

2、涉及到reward模型时，需要两个模型的tokenizer相同。

3、使用deepspeed时需要通过accelerate进行使用，直接deepspeed的话会报错(目前似乎没有很好的解决方案)

4、一般来说trl的trainer是不支持使用deepspeed的optimizer和scheduler的

5、不支持Qlora和deepspeed zero-3，支持Qlora和deepspeed zero-2

6、训练Qwen2时遇到报错，提示```no padding token is defined```。需要在qwen2 ```config.json```中添加pad_token_id,在tokenizer中设置没用。

7、PPO/RLOO参数解释：

See:https://github.com/huggingface/trl/issues/1740

The ``num_train_epochs`` and ``num_ppo_epochs`` are actually two different things. The num_train_epochs means how many epochs do we go over the dataset, the num_ppo_epochs means the number of epochs we perform PPO updates on a batch of data. So, there is a subtle but meaningful difference here.

8、CPO系列不支持fp16，支持bf16

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

具体介绍可参见文章：[知识蒸馏](https://zhuanlan.zhihu.com/p/1064724364)

### Quick Star
进入script目录下bash运行```gkd_run.sh```即可，修改对应参数运行。同样支持Deepspeed.


```bash
bash gkd_run.sh
```

**参数介绍**：
- lmbda：0时为Supervised KD，1时为GKD。可在[0,1]范围内选择，这样就会混合比例
- beta:  0时loss为KLD， 1时为JSD。可在[0,1]范围内选择，这样就会混合比例
- seq_kd: True时Supervised KD将替换为Seq KD，默认为False，其他不变。
- model_name_or_path：Student Model，即你需要训练的模型
- teacher_model_name_or_path：Teacher Model, 不训练。

## Rejected Sampling
待更新

## 感谢

特别感谢huggingface trl做出的强大贡献，通过 trl 我们真的可以很容易简洁的实现RLHF。