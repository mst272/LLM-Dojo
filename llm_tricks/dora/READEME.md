# DoRA: Weight-Decomposed Low-Rank Adaptation

此为Dora微调方法的实现(目前huggingface也已集成dora，故使用可以直接使用huggingface，本模块可以作为详细的理论学习)

```python
from peft import LoraConfig

# Initialize DoRA configuration
config = (
    use_dora=True, ...
)
```




Implementation of "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al, 2024) https://arxiv.org/pdf/2402.09353.pdf


## Tips：
Dora是基于Lora的变体，故也对Lora进行了简单的示例。


DoRA可以分两步描述，其中第一步是将预训练的权重矩阵分解为幅度向量（m）和方向矩阵（V）。第二步是将LoRA应用于方向矩阵V并单独训练幅度向量m。


## 实验结果如下：
运行 dora_example.py。超参数设置参考文件内。小模型具有局限性，具体dora和lora的实际效果对比还需要更多的实验。

```python
Epoch: 001/001 | Batch 000/938 | Loss: 2.3010
Epoch: 001/001 | Batch 400/938 | Loss: 0.4533
Epoch: 001/001 | Batch 800/938 | Loss: 0.0464
Epoch: 001/001 training accuracy: 95.31%
Time elapsed: 0.11 min
Total Training Time: 0.11 min
Test accuracy: 96.88%
Epoch: 001/002 | Batch 000/938 | Loss: 0.1734
Epoch: 001/002 | Batch 400/938 | Loss: 0.0447
Epoch: 001/002 | Batch 800/938 | Loss: 0.1270
Epoch: 001/002 training accuracy: 96.88%
Time elapsed: 0.11 min
Epoch: 002/002 | Batch 000/938 | Loss: 0.0626
Epoch: 002/002 | Batch 400/938 | Loss: 0.2149
Epoch: 002/002 | Batch 800/938 | Loss: 0.1430
Epoch: 002/002 training accuracy: 95.31%
Time elapsed: 0.23 min
Total Training Time: 0.23 min
Test accuracy LoRA finetune: 96.88%
Epoch: 001/002 | Batch 000/938 | Loss: 0.1588
Epoch: 001/002 | Batch 400/938 | Loss: 0.1235
Epoch: 001/002 | Batch 800/938 | Loss: 0.0506
Epoch: 001/002 training accuracy: 100.00%
Time elapsed: 0.11 min
Epoch: 002/002 | Batch 000/938 | Loss: 0.1374
Epoch: 002/002 | Batch 400/938 | Loss: 0.0892
Epoch: 002/002 | Batch 800/938 | Loss: 0.0606
Epoch: 002/002 training accuracy: 95.31%
Time elapsed: 0.23 min
Total Training Time: 0.23 min
Test accuracy DoRA finetune: 98.44%
```
