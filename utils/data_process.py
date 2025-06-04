import pandas as pd
import json
from torch.utils.data import Dataset
from loguru import logger
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizerBase


class MultiRoundDataProcess(Dataset):
    def __init__(self, file, tokenizer: PreTrainedTokenizerBase, max_length: int, auto_adapt=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = []  # 初始化数据列表
        self.auto_adapt = auto_adapt

        # --- 文件读取逻辑 ---
        if not os.path.exists(file):
            raise ValueError(f"路径 '{file}' 不存在")
        # logger.info(f"开始从 '{file}' 加载数据...")
        if os.path.isfile(file):
            if not file.endswith('.jsonl'):
                raise ValueError(f"文件 '{file}' 不是 JSONL 文件")
            self._read_single_file(file)
        else:
            self._read_directory(file)
        if not self.data_list:
            raise ValueError(f"未能从 '{file}' 读取到任何数据")
        # logger.info(f"从 '{file}' 加载了 {len(self.data_list)} 条数据。")
        # --- 文件读取逻辑结束 ---

    def _read_single_file(self, file_path):
        """读取单个 JSONL 文件，将每行解析为字典并添加到 data_list"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                for i, line in enumerate(f):
                    try:
                        # 直接存储解析后的字典
                        self.data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"解析文件 {file_path} 第 {i + 1} 行时发生 JSON 错误: {e}。 行内容: '{line[:100]}...'")
                        # 策略：跳过错误行 或 抛出异常中止整个加载过程
                        continue  # 选择跳过当前行
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {str(e)}")
            raise  # 抛出异常

    def _read_directory(self, dir_path):
        """读取目录中的所有 JSONL 文件"""
        jsonl_files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            raise ValueError(f"目录 '{dir_path}' 中未找到 JSONL 文件")
        # logger.info(f"在目录 '{dir_path}' 中找到 {len(jsonl_files)} 个 JSONL 文件。")
        for file_name in jsonl_files:
            file_path = os.path.join(dir_path, file_name)
            logger.info(f"正在处理文件: {file_path}")
            try:
                self._read_single_file(file_path)
            except Exception as e:
                # 记录警告并跳过该文件，继续处理其他文件
                logger.warning(f"处理文件 {file_path} 时发生错误，已跳过: {str(e)}")
                continue

    def __len__(self):
        # data_list 存储的是字典
        return len(self.data_list)

    def __getitem__(self, item) -> Optional[Dict[str, List[int]]]:  # 返回 Optional[Dict]
        # data 是已经加载的字典
        data = self.data_list[item]
        try:
            # messages = data['messages']
            messages = data.get('messages', data.get('message'))
            if self.auto_adapt:
                # 基本的数据有效性检查
                if not isinstance(messages, list) or not messages:
                    raise ValueError("无效或空的 'messages' 列表")

                # --- 标准 SFT 检查：确保最后一条消息来自助手 ---
                if messages[-1]['role'] != 'assistant':
                    # 对于不符合格式的数据，可以选择跳过或报错
                    logger.warning(
                        f"跳过第 {item} 项：最后一条消息的角色是 '{messages[-1]['role']}'，应为 'assistant'。数据: {messages}")
                    # 报错返回空
                    raise ValueError(f"第 {item} 项数据格式错误：最后一条消息角色应为 'assistant'。")
                # --- 检查结束 ---

                # --- 使用 apply_chat_template 进行分词 ---

                full_input_ids_untruncated = self.tokenizer.apply_chat_template(
                    messages,
                    add_special_tokens=False,
                    return_dict=True,
                    return_tensors=None
                )['input_ids']

                # 2. 分词提示部分 (不包括最后一条助手消息)
                if len(messages) > 1:
                    prompt_messages = messages[:-1]  # 去掉最后一条助手消息
                    # 使用 add_generation_prompt=True
                    prompt_tokenized = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        add_special_tokens=False,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors=None,
                    )
                    prompt_ids = prompt_tokenized['input_ids']
                    prompt_len = len(prompt_ids)
                else:
                    raise ValueError("无效或空的 'messages' 列表")

                # --- 步骤 3: 校验前缀 ---
                if not full_input_ids_untruncated[:prompt_len] == prompt_ids:
                    logger.error(f"第 {item} 项出现前缀不匹配错误！(优化版)")
                    logger.warning(f"跳过第 {item} 项：因前缀不匹配（可能是对话模板问题）。")
                    raise ValueError(f"跳过第 {item} 项：因前缀不匹配（可能是对话模板问题）。")
                # 提示部分为 0，助手回复部分为1
                labels_untruncated = ([0] * prompt_len) + ([1] * len(full_input_ids_untruncated[prompt_len:]))

                # --- 步骤 5: 截断---
                input_ids = full_input_ids_untruncated
                labels = labels_untruncated
                if len(input_ids) > self.max_length:
                    # 截断 labels 以匹配 input_ids 的长度 (保留前面部分)
                    input_ids = input_ids[:self.max_length]
                    labels = labels[:self.max_length]

                # --- 步骤 6: 创建 Attention Mask ---
                # Attention mask 应与最终截断后的 input_ids 长度一致，值为 1
                attention_mask = [1] * len(input_ids)

                if not (len(input_ids) == len(attention_mask) == len(labels)):
                    logger.error(f"第 {item} 项最终长度不匹配！(优化版)")
                    logger.error(f"  Input IDs len: {len(input_ids)}")
                    logger.error(f"  Attention Mask len: {len(attention_mask)}")
                    logger.error(f"  Labels len: {len(labels)}")
                    logger.warning(f"跳过第 {item} 项：因最终长度不匹配。")
                    raise ValueError(f"跳过第 {item} 项：因最终长度不匹配。")

                # --- 返回结果字典 ---
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "target_mask": labels
                }
                return inputs
            else:
                # 不使用 apply_chat_template，直接拼接内容
                input_ids = []
                target_mask = []
                for conv in messages:
                    conv_ids = self.tokenizer.encode(conv['content'], add_special_tokens=False)
                    input_ids.extend(conv_ids)
                    if conv['role'] == 'assistant':
                        target_mask.extend([1] * len(conv_ids))
                    else:
                        target_mask.extend([0] * len(conv_ids))
                attention_mask = [1] * len(input_ids)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "target_mask": target_mask
                }
                return inputs

        except Exception as e:
            # 捕获其他意外错误
            logger.exception(f"处理第 {item} 项时发生意外错误。 数据: {data}。 错误: {e}")
            # 根据策略返回 None 或重新抛出异常
            inputs = {
                "input_ids": None,
                "attention_mask": None,
                "target_mask": None
            }  # 返回 None 以允许训练在过滤后继续
            return inputs


# class MultiRoundDataProcess1(Dataset):
#     def __init__(self, file, tokenizer, max_length, auto_adapt=True):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.data_list = []
#         self.auto_adapt = auto_adapt
#
#         # 检查路径是否存在
#         if not os.path.exists(file):
#             raise ValueError(f"路径 '{file}' 不存在")
#
#         # 判断输入路径是文件还是目录
#         if os.path.isfile(file):
#             # 如果是单个文件，直接读取
#             if not file.endswith('.jsonl'):
#                 raise ValueError(f"文件 '{file}' 不是JSONL文件")
#             self._read_single_file(file)
#         else:
#             # 如果是目录，读取所有JSONL文件
#             self._read_directory(file)
#         if not self.data_list:
#             raise ValueError("没有读取到任何数据")
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def _read_single_file(self, file_path):
#         """读取单个JSONL文件"""
#         try:
#             with open(file_path, 'r', encoding='utf8') as f:
#                 self.data_list.extend(f.readlines())
#         except Exception as e:
#             # print(f"读取文件 {file_path} 时发生错误: {str(e)}")
#             raise
#
#     def _read_directory(self, dir_path):
#         """读取目录中的所有JSONL文件"""
#         jsonl_files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl')]
#
#         if not jsonl_files:
#             raise ValueError(f"目录 '{dir_path}' 中没有找到JSONL文件")
#
#         for file_name in jsonl_files:
#             file_path = os.path.join(dir_path, file_name)
#             try:
#                 self._read_single_file(file_path)
#             except Exception as e:
#                 # print(f"处理文件 {file_path} 时发生错误: {str(e)}")
#                 continue
#
#     def __getitem__(self, item):
#         # 开始自动判断并适配chat template
#         data = self.data_list[item]
#         data = json.loads(data)
#         # message = data['message']
#         # fix, message will be removed later
#         message = data.get('messages', data.get('message'))
#
#         input_ids = []
#         target_mask = []
#
#         if self.auto_adapt:
#             # 使用 apply_chat_template 生成格式化文本
#             text = self.tokenizer.apply_chat_template(message, tokenize=False)
#             # 对整个文本进行分词
#             input_ids = self.tokenizer.encode(text, add_special_tokens=False)
#             # 初始化 target_mask 为全 0
#             target_mask = [0] * len(input_ids)
#             # 记录当前处理的 token 位置
#             current_position = 0
#             for conv in message:
#                 if conv['role'] == 'assistant':
#                     # 对助手消息内容进行分词
#                     assistant_ids = self.tokenizer.encode(conv['content'], add_special_tokens=False)
#                     # 在 input_ids[current_position:] 中查找 assistant_ids 的起始位置
#                     position = find_sublist_start(input_ids[current_position:], assistant_ids)
#                     if position == -1:
#                         raise ValueError("Assistant message not found in input_ids")
#                     # 计算在整个 input_ids 中的实际位置
#                     actual_position = current_position + position
#                     assistant_len = len(assistant_ids)
#                     # 将 target_mask 中对应位置设置为 1
#                     target_mask[actual_position:actual_position + assistant_len] = [1] * assistant_len
#                     # 找到助手消息后的 EOS token 位置
#                     eos_position = actual_position + assistant_len
#                     if eos_position < len(input_ids):
#                         # 检查下一个token是否为EOS token
#                         next_token = input_ids[eos_position]
#                         if next_token == self.tokenizer.eos_token_id:
#                             # 将EOS token的target mask设为1
#                             target_mask[eos_position] = 1
#                             # 更新 current_position 到EOS token之后
#                             current_position = eos_position + 1
#                         else:
#                             # 如果下一个不是EOS token，只更新到assistant内容之后
#                             current_position = eos_position
#         else:
#             # 不使用 apply_chat_template，直接拼接内容
#             for conv in message:
#                 conv_ids = self.tokenizer.encode(conv['content'], add_special_tokens=False)
#                 input_ids.extend(conv_ids)
#                 if conv['role'] == 'assistant':
#                     target_mask.extend([1] * len(conv_ids))
#                 else:
#                     target_mask.extend([0] * len(conv_ids))
#
#         # 对长度进行截断
#         input_ids = input_ids[:self.max_length]
#         target_mask = target_mask[:self.max_length]
#         attention_mask = [1] * len(input_ids)
#
#         # 断言长度相等
#         assert len(input_ids) == len(target_mask) == len(attention_mask)
#
#         inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "target_mask": target_mask
#         }
#
#         return inputs


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
