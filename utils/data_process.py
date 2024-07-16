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


class CommonSingleRoundDataProcess(Dataset):
    """
    默认使用的单轮对话常规dataset
    """

    def __init__(self, file, tokenizer, max_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_length = max_length

        logger.info(f'Loading data: {file}')
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info(f"There are {len(data_list)} data in dataset")
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 开始拼接每条数据
        data = self.data_list[item]
        data = json.loads(data)
        input_ids, target_mask = [], []

        if self.system_format is not None:
            system = self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)
        instruction = data['instruction']
        output = data['output']

        instruction_text = self.user_format.format(content=instruction, stop_token=self.tokenizer.eos_token)
        output_text = self.assistant_format.format(content=output, stop_token=self.tokenizer.eos_token)

        input_tokens = self.tokenizer.encode(instruction_text, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        # 判断一下输入和掩码长度是否相等
        assert len(input_ids) == len(target_mask)

        # 对长度进行截断
        input_ids = input_ids[:self.max_length]
        target_mask = target_mask[:self.max_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask
        }

        return inputs


class DpoDataset(Dataset):
    """
    单轮DpoDataset
    """
    def __init__(self, file, tokenizer, max_length, max_prompt_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)  # 将json格式转换为python字典
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        # 对prompt进行编码
        prompt = self.user_format.format(content=prompt, stop_token=self.tokenizer.eos_token)
        if self.system_format is not None:
            system = self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                prompt_input_ids = input_ids + self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)



        # 进行回答的input id编码
        chosen = self.assistant_format.format(content=chosen, stop_token=self.tokenizer.eos_token)
        rejected = self.assistant_format.format(content=rejected, stop_token=self.tokenizer.eos_token)

        chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        # 对最大长度进行截断
        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        # keep end 对prompt截断
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            max_prompt_length = max(self.max_prompt_length, self.max_seq_length - longer_response_length)
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        # 如果还不符合则回答截断
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            chosen_input_ids = chosen_input_ids[: self.max_seq_length - len(prompt_input_ids)]
            rejected_input_ids = rejected_input_ids[: self.max_seq_length - len(prompt_input_ids)]

        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids
        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1] * len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1] * len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1] * len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    # 适配DPOTrainer的接口
    def map(self, func, **kwargs):
        return self
