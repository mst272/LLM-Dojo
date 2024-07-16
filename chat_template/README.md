# 🍻Chat Template

有了transfomers框架后我们的微调工作都变得简单了起来，我们在自己进行一些模型的微调时主要的操作就是构建自己的数据集及数据格式，不同模型的在训练时都使用了不同的template。

因此，我从官方参考或实现中收集了几种主流模型的官方模板，都包含在此文档中，以供大家自己进行微调时参考。

如果您有兴趣提供更多模型聊天模板，请随时打开一个pull request或issue。

## Supported Template Model

- [Qwen](#qwen)
- [Yi](#yi)
- [Gemma](#gemma)
- [Phi-3](#phi-3)
- [DeepSeek](#deepseek)
- [Mistral](#mistral)
- [llama2](#llama2)
- [llama3](#llama3)
- [MiniCPM](#minicpm)
- [DeepSeek-coder](#deepseek-coder)

## 说明
 不同模型在是否存在默认system message上有所不同(大多数模型都是没有的)，所以以官方是否有为标准。
 
每个模型都附上了有system版本和无system版本，如果在自己训练模型的时候希望加上system message可以参照template模板自行添加。

**(建议的做法是：如果官方没提供默认system message，可直接使用无system版本)**

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
官方版本没有默认的system message，可以与llama一样不加system message使用，有
```text
<|im_start|>system
This is a system message<|im_end|>
<|im_start|>user
This is a instruction<|im_end|>
<|im_start|>assistant
This is a answer<|im_end|>
```
- 无system模式
```text
<|im_start|>user
This is a instruction<|im_end|>
<|im_start|>assistant
This is a answer<|im_end|>
```

### Gemma
官方版本不支持system

- 无system模式
```text
<bos><start_of_turn>user
This is a instruction<end_of_turn>
<start_of_turn>model
This is a answer<end_of_turn>
```

### Phi-3
官方版本没有默认的system message, 有此需求可依据下述模板自己构建

```text
<s><|system|>
This is a system message<|end|>
<|user|>
This is a instruction<end>
<|assistant|>
This is a answer<end>
```
- 无system模式
```text
<s><|user|>
This is a instruction<end>
<|assistant|>
This is a answer<end>
```

### Deepseek
官方同样没有提供默认system message，有此需求可依据下述模板自己构建
```text
<｜begin▁of▁sentence｜>This is a system message
User:This is a instruction
Assistant:This is a answer<｜end▁of▁sentence｜>
```

- 无system模式
```text
<｜begin▁of▁sentence｜>User:This is a instruction
Assistant:This is a answer<｜end▁of▁sentence｜>
```

### Mistral
没有提供system模式
- 无system模式
```text
<s>[INST]:This is a instruction [/INST]This is a answer</s>
```

### Llama2

```text
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST] This is a answer</s>
```

### Llama3
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

This is a system prompt.<|eot_id|><|start_header_id|>user<|end_header_id|>

This is the first user input.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

This is the first assistant response.<|eot_id|>
```

### MiniCPM
```text
<用户>This is a system message<AI>This is a instruction</s>
```

### DeepSeek-coder
```text
<｜begin▁of▁sentence｜>User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```
You can also add an optional system message:
```text
<｜begin▁of▁sentence｜>{system_message}

User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```