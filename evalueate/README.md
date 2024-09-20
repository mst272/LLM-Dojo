# Code Evaluation

面向代码大模型评测，目前支持的有humaneval

注意：
只能在Linux机器上进行，windows上 execution 部分有错误


生成文件为jsonl形式，前面保持不变，新增字段如下：

output:模型接收prompt后产生的原本输出
generation: 经过提取代码部分及添加测试后的输出
