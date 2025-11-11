# Code Evaluation

面向代码大模型评测，目前支持的有humaneval，后续会逐步新增一些。

注意：
只能在Linux机器上进行，windows上 execution 部分有错误


## Quick Start

evaluate文件下的run.sh作为一个启动示例，详细参数解释可见args.py。

其中评测集数据应为jsonl格式
```bash
bash run.sh
```

### 评测生成文件

模型评测完成后主要生成三个文件：

1、out.jsonl: 模型输出，在评测数据的基础上新增字段如下：
- output:模型接收prompt后产生的原本输出
- generation: 经过提取代码部分及添加测试后的输出

2、logs.jsonl: 评测测试用例运行的信息

3、metric.json:  评测结果指标

## 新增评测
如若想要新增评测任务，可以继承base_utils中的类进行相关设置，然后在task文件夹下创建相关文件进行继承。
最后在main.py文件的TASKS中添加即可。