# 从零实现强化学习DPO（SimPO）训练代码

## Quick start
```python
python dpo_train.py
```

## 说明
本文档下的从零实现只是一个学习的demo，用以理解原理所用，并没有增加分布式等。所以尽管使用2B的小模型，显存占用也高达30+GB。

精度设置fp16可能会出现loss 为nan的现象

```dpo_train.py```为训练主路径， 相关loss计算在```loss.py```.

如果想要使用DPO或者Simpo、CPO等强化学习方法真正训练的话，
可以使用本项目中的rlhf构建的强化学习框架：[RLHF](../../rlhf/README.md)

支持deepspeed的单机多卡Lora、Dora、Qlora、全量参数训练，并自动适配模型的chat template。