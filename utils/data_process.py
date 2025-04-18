import json
from torch.utils.data import Dataset
from loguru import logger
from pathlib import Path
import pandas as pd
import os
from typing import Dict, List, Tuple


class MultiRoundDataProcess(Dataset):
    def __init__(self, file, tokenizer, max_length, auto_adapt=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # logger.info(f'Loading data: {file}')
        # with open(file, 'r', encoding='utf8') as f:
        #     data_list = f.readlines()
        # # logger.info(f"There are {len(data_list)} data in dataset")
        # self.data_list = data_list
        self.data_list = []
        self.auto_adapt = auto_adapt

        # 检查路径是否存在
        if not os.path.exists(file):
            raise ValueError(f"路径 '{file}' 不存在")

        # 判断输入路径是文件还是目录
        if os.path.isfile(file):
            # 如果是单个文件，直接读取
            if not file.endswith('.jsonl'):
                raise ValueError(f"文件 '{file}' 不是JSONL文件")
            self._read_single_file(file)
        else:
            # 如果是目录，读取所有JSONL文件
            self._read_directory(file)
        if not self.data_list:
            raise ValueError("没有读取到任何数据")

    def __len__(self):
        return len(self.data_list)

    def _read_single_file(self, file_path):
        """读取单个JSONL文件"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                self.data_list.extend(f.readlines())
        except Exception as e:
            # print(f"读取文件 {file_path} 时发生错误: {str(e)}")
            raise

    def _read_directory(self, dir_path):
        """读取目录中的所有JSONL文件"""
        jsonl_files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl')]

        if not jsonl_files:
            raise ValueError(f"目录 '{dir_path}' 中没有找到JSONL文件")

        for file_name in jsonl_files:
            file_path = os.path.join(dir_path, file_name)
            try:
                self._read_single_file(file_path)
            except Exception as e:
                # print(f"处理文件 {file_path} 时发生错误: {str(e)}")
                continue

    def __getitem__(self, item):
        # 开始自动判断并适配chat template
        data = self.data_list[item]
        data = json.loads(data)
        # message = data['message']
        # fix, message will be removed later
        message = data.get('messages', data.get('message'))

        input_ids = []
        target_mask = []

        if self.auto_adapt:
            # 使用 apply_chat_template 生成格式化文本
            text = self.tokenizer.apply_chat_template(message, tokenize=False)
            # 对整个文本进行分词
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            # 初始化 target_mask 为全 0
            target_mask = [0] * len(input_ids)
            # 记录当前处理的 token 位置
            current_position = 0
            for conv in message:
                if conv['role'] == 'assistant':
                    # 对助手消息内容进行分词
                    assistant_ids = self.tokenizer.encode(conv['content'], add_special_tokens=False)
                    # 在 input_ids[current_position:] 中查找 assistant_ids 的起始位置
                    position = find_sublist_start(input_ids[current_position:], assistant_ids)
                    if position == -1:
                        raise ValueError("Assistant message not found in input_ids")
                    # 计算在整个 input_ids 中的实际位置
                    actual_position = current_position + position
                    assistant_len = len(assistant_ids)
                    # 将 target_mask 中对应位置设置为 1
                    target_mask[actual_position:actual_position + assistant_len] = [1] * assistant_len
                    # 找到助手消息后的 EOS token 位置
                    eos_position = actual_position + assistant_len
                    if eos_position < len(input_ids):
                        # 检查下一个token是否为EOS token
                        next_token = input_ids[eos_position]
                        if next_token == self.tokenizer.eos_token_id:
                            # 将EOS token的target mask设为1
                            target_mask[eos_position] = 1
                            # 更新 current_position 到EOS token之后
                            current_position = eos_position + 1
                        else:
                            # 如果下一个不是EOS token，只更新到assistant内容之后
                            current_position = eos_position
        else:
            # 不使用 apply_chat_template，直接拼接内容
            for conv in message:
                conv_ids = self.tokenizer.encode(conv['content'], add_special_tokens=False)
                input_ids.extend(conv_ids)
                if conv['role'] == 'assistant':
                    target_mask.extend([1] * len(conv_ids))
                else:
                    target_mask.extend([0] * len(conv_ids))

        # 对长度进行截断
        input_ids = input_ids[:self.max_length]
        target_mask = target_mask[:self.max_length]
        attention_mask = [1] * len(input_ids)

        # 断言长度相等
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


def find_sublist_start(main_list, sub_list):
    """
    find_sublist_start

    Args:
    main_list (list)
    sub_list (list)
    """
    sub_len = len(sub_list)
    main_len = len(main_list)
    for i in range(main_len - sub_len + 1):
        if main_list[i:i + sub_len] == sub_list:
            return i
        # 因为会有开头decode不一样的情况出现
        elif main_list[i + 1:i + sub_len] == sub_list[1:]:
            return i
    return -1


# datasets直接加载也可以，但考虑可能的错误自己构建一个作为备选方案
# [{'content': [{'index': 0, 'text': None, 'type': 'image'},
#    {'index': None,
#     'text': '\nWhat may be the purpose of this gathering in the field?',
#     'type': 'text'}],
#   'role': 'user'},
#  {'content': [{'index': None,
#     'text': 'The purpose engaging in an outdoor activity.',
#     'type': 'text'}],
#   'role': 'assistant'}]


# 下面是我们的规定数据格式
# [{'question':'Who is the person in the picture?', 'answer':'this is panda'},{'question':'Is it cool?', 'answer':'Yes'}]


class VlmQaDataset(Dataset):
    def __init__(self, data_file):
        self.data_file = Path(data_file)
        metadata_path = self.data_file.joinpath("metadata.jsonl")

        self.messages = pd.read_json(metadata_path, lines=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item) -> Tuple[List[str], List[str], Path]:
        message = self.messages['messages'][item]

        # todo: 单图片输入，后续可扩展多图片输入？
        image = self.messages['file_name'][item]
        image_path = self.data_file.joinpath(image)

        questions = []
        answers = []

        for text in message:
            questions.append(text['question'])
            answers.append(text['answer'])

        return questions, answers, image_path
