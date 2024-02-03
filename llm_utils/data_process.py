import json

from torch.utils.data import Dataset
from loguru import logger


class QwenDataProcess(Dataset):
    """
    Qwen 相关模型的数据处理格式
    """

    def __init__(self, file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.im_start_id = tokenizer.im_start_id
        self.im_end_id = tokenizer.im_end_id
        self.enter_token_ids = tokenizer.encode('\n')  # 表示回车键
        self.max_len = max_len
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        """
        数据拼接格式如下：
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        你好呀<|im_end|>
        <|im_start|>assistant
        你好，我是xxx，很高兴为您服务<|im_end|>
        """
        data = self.data_list[item]
        data = json.loads(data)

        # 数据部分的instruction输入
        instruction_text = f'<|im_start|>system\n{data["instruction"]}<|im_end|>\n'
        instruction_ids = self.tokenizer.encode(instruction_text, add_special_tokens=False)
        target_mask = [0] * len(instruction_ids)

        # 单轮对话
        input_text = data['input'].strip()
        output_text = data['output'].strip()

        input_tokens = self.tokenizer.encode(f'<|im_start|>user\n{input_text}<|im_end|>\n', add_special_tokens=False)
        output_tokens = self.tokenizer.encode(f'<|im_start|>assistant\n{output_text}<|im_end|>\n',
                                              add_special_tokens=False)

        input_ids = instruction_ids + input_tokens + output_tokens
        # input_tokens部分不计算loss
        target_mask += [0] * len(input_tokens)
        # '<|im_start|>assistant\n'占3个token，结尾的'<|im_end|>'占1个token，不计算它们的loss
        target_mask += [0] * 3 + [1] * (len(output_tokens) - 4) + [0]

        assert len(input_ids) == len(target_mask), 'input_ids与target_mask长度出现错误'

        # 设置范围为max_len长度
        input_ids = input_ids[:self.max_len]
        target_mask = target_mask[:self.max_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_mask
        }

        return inputs


class YiDataProcess(QwenDataProcess):
    """
    Yi 相关模型的数据处理格式
    与Qwen相同
    """
    pass
