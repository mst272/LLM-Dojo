# RLHF全流程

最近一直在从零构建强化学习的全套流程，特别感谢huggingface trl做出的强大贡献，通过trl 我们真的可以很容易简洁的实现RLHF


主要资源是在1-3张40G A100上进行实验，其中需要很多显存优化策略，踩了很多坑，包括deepspeed、unsloth等的兼容性问题。

包括：
- Reward模型的训练
- RLOO、DPO、PPO、SimPO等多种变体


## Step1 训练Reward Model

第一步就是需要训练一个合格的奖励模型。这一步还是比较简单的，且也不用占用过多的显存。

## Step2 RL：基于不同优化方法进行强化学习，如DPO、PPO等

PPO：目前zero3训练还有报错，暂未查明原因



一般来说trl的trainer是不支持使用deepspeed的optimizer和scheduler的



### 多卡训练注意
使用deepspeed时最好通过accelerate进行使用，直接deepspeed的话会报错(目前似乎没有很好的解决方案)

#### 建议方式
所以使用zero-3的accelerate命令如下：
```bash
CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --config_file ./deepspeed_zero3.yaml rloo_train2.py
```
- CUDA_VISIBLE_DEVICES：代表你要用的卡，可以指定多块，但是要在deepspeed_zero3.yaml文件中修改```num_processes```为对应数量
- config_file: deepspeed的yaml文件路径，可以支持zero1/2/3

#### 不建议方式
直接使用deepspeed命令训练的话只能使用策略zero-2及以下，**zero-3是无法使用的**。在config中指定zero文件
```bash
deepspeed --include localhost:0 rloo_train2.py
```