# DoRA: Weight-Decomposed Low-Rank Adaptation

此为Dora微调方法的实现

Implementation of "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al, 2024) https://arxiv.org/pdf/2402.09353.pdf


## Tips：
Dora是基于Lora的变体，故也对Lora进行了简单的示例。


DoRA可以分两步描述，其中第一步是将预训练的权重矩阵分解为幅度向量（m）和方向矩阵（V）。第二步是将LoRA应用于方向矩阵V并单独训练幅度向量m。