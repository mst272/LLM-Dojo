# Train with prediction eval

详细的使用说明及函数文档待更新，目前只是一个dirty的实现。



## test data

评测数据格式为jsonl，以代码test为例，具体可见data/test.jsonl。
包含两个字段：
- prompt: test 的问题
- label：答案，代码来说即是测试用例

## Quick start

使用vllm进行生成，其余卡进行训练。

启动脚本位置：utils/eval/vllm/run_serve.sh

1、启动vllm_serve，例如使用2卡

```shell
bash run_serve.sh
```

2、开启训练

启动脚本位置：run_eval_test.sh

```shell
bash run_eval_test.sh --eval
```
运行脚本，注意要跟--eval，一些参数配置可参考run_eval_test.sh文件。





## Tip

wandb出问题可以尝试：
pip install wandb==0.12.18

可能出现的问题：

1、直接deepspeed --master_port 29508 --include localhost:2,3,4,5,6,7 main_train.py保存checkpoint时有问题，所以建议
accelerate launch --config_file rlhf/ds_config/ds_zero3.yaml main_train.py


2、训练时出现训推不一致问题，训练中评测跟保存后结果对不上，最后找到原因是因为没有enable_prefix_caching=False。
不过尝试之后仍然会有偏差，但是影响不大，曲线的轨迹是可以反映模型在测试集上的效果的。

待验证：可能是由于dropout层的原因，后续计划禁止dropout尝试


参考：https://github.com/huggingface/open-r1/issues/433



## Reference
最后，代码借鉴了trl项目，感谢trl为开源做出的贡献。