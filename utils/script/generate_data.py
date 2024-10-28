import pandas as pd

ORIGINAL_DATA_PATH = '../../1.jsonl'  # 原数据路径
OUT_DATA_PATH = './out_data.jsonl'  # 转换为框架适用的role content模式的输出路径


data1 = pd.read_json(ORIGINAL_DATA_PATH, lines=True)
# 创建一个空的列表来存储处理后的数据
processed_data = []
# 迭代每一行数据
for index, row in data1.iterrows():
    message = [
        {"role": "user", "content": row['instruction']},
        {"role": "assistant", "content": row['output']}
    ]
    processed_data.append({"message": message})
# 将处理后的数据转换为 DataFrame
processed_df = pd.DataFrame(processed_data)
# 保存为jsonl格式
processed_df.to_json(OUT_DATA_PATH, orient='records', lines=True, force_ascii=False)
