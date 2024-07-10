# RLHF全流程

最近一直在从零构建强化学习的全套流程，特别感谢huggingface trl做出的强大贡献，通过trl 我们真的可以很容易简洁的实现RLHF


主要资源是在1-3张40G A100上进行实验，其中需要很多显存优化策略，踩了很多坑，包括deepspeed、unsloth等的兼容性问题。

包括：
- Reward模型的训练
- RLOO、DPO、PPO、SimPO等多种变体


## Step1 训练Reward Model

第一步就是需要训练一个合格的奖励模型。这一步还是比较简单的，且也不用占用过多的显存。

**对于Reward模型，需要自己去看AutoModelForSequenceClassification是否可以加载其Classification模型，不能的话需要在其config文件中映射。**

## Step2 RL：基于不同优化方法进行强化学习，如DPO、PPO等

PPO：目前zero3训练还有报错，暂未查明原因



一般来说trl的trainer是不支持使用deepspeed的optimizer和scheduler的



### 多卡训练注意
使用deepspeed时最好通过accelerate进行使用，直接deepspeed的话会报错(目前似乎没有很好的解决方案)

#### 建议方式
所以使用zero-3的accelerate命令如下()：
```bash
CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --config_file ./deepspeed_zero3.yaml rloo_train2.py
```
- CUDA_VISIBLE_DEVICES：代表你要用的卡，可以指定多块，但是要在deepspeed_zero3.yaml文件中修改```num_processes```为对应数量
- config_file: deepspeed的yaml文件路径，可以支持zero1/2/3



DPO终于也可以了，但是跟deepspeed适配还是有些问题，目前A100 40GB只能训2B的模型。因为zero3 offload会报莫名错误(即只能在非offload情况下训练，所以显存占用很高)，后续还需探讨如何优化。


ds.yaml文件中main_process_port如果被占用则加一个数字即可。错误如下：

> ConnectionError: Tried to launch distributed communication on port `29500`, but another process is utilizing it. Please specify a different port (such as using the `--main_process_port` flag or specifying a different `main_process_port` in your config file) and rerun your script. To automatically use the next open port (on a single node), you can set this to `0`.



### 支持矩阵
✅ 代表支持deepspeed 全策略

| 支持方法/deepspeed | LORA | QLORA | Full | Unsloth(待更新) |
|----------------|------|-------|------|--------------|
| RLOO           | ✅    | Zero2 | ✅    | ❌            |
| PPO            |      |       |      | ❌            |
| SimPO          |      |       |      | ❌            |




### 显存实验
res——length为64

| **RLHF** | **deepspeed**   | **方式** | **Reward Model** | **SFT Model** | **显存占用**              |
|----------|-----------------|--------|------------------|---------------|-----------------------|
| RLOO     | Zero 3 cpu  cpu | Lora   | QWEN2(7B)        | QWEN2(7B)     | 2 x A100(40GB):15~30G |
| RLOO     | Zero 3 cpu  cpu | Full   | QWEN2(7B)        | QWEN2(7B)     | 2 x A100(40GB):速度很慢   |
| RLOO     | Zero 2 cpu      | Qlora  | QWEN2(7B)        | QWEN2(7B)     | 2 x A100(40GB):30~40G |




注：
#### RLOO：

**RLOO 支持zero-3的offload_param，支持offload_optimizer**可见deepspeed_zero3.yaml示例.

不支持Qlora和deepspeed zero-3：可能需要和get_peft_model才不会报错。

deepspeed zero-3 支持Lora

QWEN应该挺大，用deepseek 6.7B好一些
   
RLOO:  R:2B   S:7B
                          optim  param
accelerate 命令 LORA zero3  cpu     cpu ，res_length 64 : 成功  (30G内)  双卡A100(40G)

accelerate 命令 QLORA zero3  cpu     cpu ，res_length 64 : 报错TypeError: output tensor must have the same type as input tensor 

accelerate 命令 QLORA zero3  none    none ，res_length 64 : 报错TypeError: output tensor must have the same type as input tensor  故无关


accelerate 命令 QLORA zero3  none    none ，res_length 64   单卡可以。 即单卡支持QLORA

破案了，QLora 只支持zero2及以下，不支持zero3。    zero2，zero3理论上都支持两个cpu     cpu。


需要确定一下QLORA时  policy和ref是否都需要量化，可能会报bit冲突的错误。。试了一下  单模型QLORA是可以的。

#### PPO：

R:2B   S:6.7B
                          optim  param
accelerate 命令 LORA zero3  cpu     cpu ，res_length 64 :   失败  ，应该不支持zero3的param offload策略

accelerate 命令 LORA zero2  cpu         ，res_length 64 :  OOM  双卡A100

accelerate 命令 FULL zero2  cpu         ，res_length 64 :  OOM  双卡A100

accelerate 命令 LORA zero2  cpu         ，res_length 64 :  OOM 三卡A100

accelerate 命令 LORA zero3  cpu         ，res_length 64 :  报错 不支持zero3的optim offload 三卡A100

accelerate 命令 LORA zero3  none   none     ，res_length 64 : oom

初步结论只能结合ZERO-2进行 PPO的训练。

accelerate 命令 QLORA only policy zero2  cpu         ，res_length 64 :  报错 看来zero2的 optim offload 也不支持 双卡A100


上述可能是数据集有问题，增大长度重新测试。

accelerate 命令 LORA zero3  cpu     cpu ，res_length 64 :   成功  且显存占用不高20G左右    双卡A100

accelerate 命令 QLORA only policy zero3  cpu     cpu ，res_length 64 : 确实不支持zero3 + qlora， 报错如下 TypeErrorTypeError: : output tensor must have the same type as input tensoroutput tensor must have the same type as input tensor

accelerate 命令 QLORA only policy zero2  cpu         ，res_length 64 :  成功  30GB左右   双卡A100

accelerate 命令 FULL zero3  cpu   cpu      ，res_length 64 :   双卡A100