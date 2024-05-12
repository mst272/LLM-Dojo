# 🍻Chat Template

有了transfomers框架后我们的微调工作都变得简单了起来，我们在自己进行一些模型的微调时主要的操作就是构建自己的数据集及数据格式，不同模型的在训练时都使用了不同的template。

因此，我从官方参考或实现中收集了几种主流模型的官方模板，都包含在此文档中，以供大家自己进行微调时参考。

如果您有兴趣提供更多模型聊天模板，请随时打开一个pull request或issue。

## Supported Template Model

- [Qwen](#qwen)
- [Yi](#yi)
- [Gemma](#gemma)

## 说明
 不同模型有的有默认的system message，有的没有，所以以官方是否有为标准。如果在自己训练模型的时候希望加上system message可以参照template模板自行添加

### Qwen
官方版本默认的system message即：You are a helpful assistant
```text
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
This is a instruction<|im_end|>
<|im_start|>assistant
This is a answer<|im_end|>
```

### Yi
官方版本没有默认的system message，可以与llama一样不加system message使用
```text
<|im_start|>user
This is a instruction<|im_end|>
<|im_start|>assistant
This is a answer<|im_end|>
```

### Gemma
官方版本没有默认的system message, 有此需求可依据下述模板自己构建
```text
<bos><start_of_turn>user
This is a instruction<end_of_turn>
<start_of_turn>model
This is a answer<end_of_turn>
```

### Phi-3
官方版本没有默认的system message, 有此需求可依据下述模板自己构建
```text
<s>user
This is a instruction<end>
<|assistant|>
This is a answer<end>
```

### Deepseek
官方同样没有提供默认system message，若有自己的system message可按下述构建
```text
<｜begin▁of▁sentence｜>This is a system message
User:This is a instruction
Assistant:This is a answer<｜end▁of▁sentence｜>
```