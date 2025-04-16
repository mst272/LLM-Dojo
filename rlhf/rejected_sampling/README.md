# Rejected Sampling

## 1、Generate

### Data format

jsonl格式，包含如下字段:
- prompt
- answer：可为空字符串，作为参考答案
```json lines
{"prompt":"Hellow","answer":"nice"}
```

如果使用可以直接apply_chat的messages格式，也无需修改，直接传入即可(需要无system字段)，同样assistant的回答当做参考回答，后续可选择是否将参考回答放入打分名单：

```json lines
{"message": [{"role": "user", "content": "How many helicopters can a human eat in one sitting"},{"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together"}]}
```

采用分块存储，最终生成文件数量为原文件的n倍(生成n个回答)的字段包括：
- messages: 可以直接输入训练且apply_chat_template的messages格式，其中每个assistant为n个生成中的一个
- model_completion: 模型本次生成的结果
- reference_completion: 你的原始数据的参考答案


## 2、Rejected sampling评测阶段

目前只支持通过api进行评测选择，传统的classification模型几乎用不到了，所以就进行了去除。

参数待配置，目前还只是一个草案
