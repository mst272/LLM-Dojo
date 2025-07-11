import pandas as pd
import json
from torch.utils.data import Dataset
from loguru import logger
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizerBase


class MultiRoundDataProcess(Dataset):
    """
    支持读取：
      1. 单个 .jsonl / .parquet
      2. 目录下若干 .jsonl / .parquet 文件
    读取后，将每条记录（dict）存入 self.data_list

    auto_adapt为True时通过apply_chat自动适配单轮和多轮的labels，False时不进行apply_chat，直接拼接内容。(Qwen3多轮暂不支持auto_adapt，单轮可以)
    """

    def __init__(self, file_or_dir, tokenizer: PreTrainedTokenizerBase, max_length: int, auto_adapt=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = []  # 初始化数据列表
        self.auto_adapt = auto_adapt

        # —— 新增：识别是否 Qwen-3 —— #
        self.is_qwen3 = self.is_qwen3_tokenizer(tokenizer)

        # 文件读取逻辑
        if not os.path.exists(file_or_dir):
            raise ValueError(f"路径 '{file_or_dir}' 不存在")

        if os.path.isfile(file_or_dir):
            self._dispatch_single(file_or_dir)
        else:
            self._read_directory(file_or_dir)

        if not self.data_list:
            raise ValueError(f"未能从 '{file_or_dir}' 读取到任何数据")

    SUPPORTED_EXTS = {".jsonl", ".parquet"}

    def _dispatch_single(self, file_path: str):
        """根据扩展名将单文件分发到对应读取函数"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(f"不支持的文件类型 '{ext}' (仅支持 .jsonl / .parquet)")
        if ext == ".jsonl":
            self._read_single_jsonl(file_path)
        else:  # .parquet
            self._read_single_parquet(file_path)

    # 读取单个jsonl文件
    def _read_single_jsonl(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        self.data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"[JSONL] 解析 {file_path}:{i} 失败 -> {e}.  行内容: '{line[:100]}...'"
                        )
        except Exception as e:
            logger.error(f"[JSONL] 读取文件 {file_path} 失败: {e}")
            raise

    def _read_single_parquet(self, file_path: str):
        """
        将 parquet 的每一行转成 dict，并追加到 data_list
        """
        try:
            df = pd.read_parquet(file_path)  # DataFrame → List[dict]
            records = df.to_dict(orient="records")
            if not isinstance(records, list):
                raise ValueError("读取 parquet 后未得到记录列表")
            self.data_list.extend(records)
        except Exception as e:
            logger.error(f"[Parquet] 读取文件 {file_path} 失败: {e}")
            raise

    def _read_directory(self, dir_path: str):
        """
        遍历目录，读取所有 .jsonl / .parquet
        """
        files = [
            f
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTS
        ]
        if not files:
            raise ValueError(f"目录 '{dir_path}' 中未找到 .jsonl / .parquet 文件")

        for file_name in files:
            path = os.path.join(dir_path, file_name)
            logger.info(f"正在处理文件: {path}")
            try:
                self._dispatch_single(path)
            except Exception as e:
                logger.warning(f"跳过文件 {path}，原因: {e}")

    @staticmethod
    def is_qwen3_tokenizer(tokenizer) -> bool:
        """根据tokenizer.__class__.__name__词表大小判断是否 Qwen-3的tokenizer"""
        name = tokenizer.__class__.__name__
        vocab_size = len(tokenizer.get_vocab())
        return 'Qwen2TokenizerFast' == name and vocab_size == 151669

    @staticmethod
    def fix_qwen3_labels(input_ids: List[int], labels: List[int]) -> None:
        """
        若为 Qwen-3，将特殊 pattern 的 labels 置 0（就地修改）
        即将assistant下面的<think>\n</think>\n不计入loss
        """
        pat = [151667, 271, 151668, 271]
        plen = len(pat)
        limit = len(input_ids) - plen + 1
        for i in range(limit):
            if input_ids[i:i + plen] == pat:
                labels[i:i + plen] = [0] * plen

    def __len__(self):
        # data_list 存储的是字典
        return len(self.data_list)

    def __getitem__(self, item) -> Optional[Dict[str, List[int]]]:  # 返回 Optional[Dict]
        # data 是已经加载的字典
        data = self.data_list[item]
        try:
            messages = data.get('messages', data.get('message'))
            # 自动适配chat-template
            if self.auto_adapt:
                # 对于qwen3，去除掉数据中的system
                if self.is_qwen3:
                    if any(m.get("role") == "system" for m in messages):
                        # 保留非-system 消息，顺序不变
                        messages = [m for m in messages if m.get("role") != "system"]

                # --- 开始: 自动单轮多轮label设置的逻辑 ---
                # 1、获取完整的 token IDs
                full_input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                    return_dict=True,
                    return_tensors=None
                )['input_ids']

                # 2、迭代构建 labels (target_mask)
                generated_labels = []
                len_of_tokenized_so_far = 0

                # 逐轮遍历消息，以确定每部分对应的标签
                for i, msg in enumerate(messages):
                    role = msg.get('role')
                    if role in ['system', 'user']:
                        # 编码到当前轮次为止的对话
                        current_turn_ids = self.tokenizer.apply_chat_template(
                            messages[:i + 1],
                            add_generation_prompt=True,
                            return_dict=True,
                            add_special_tokens=False,
                        )['input_ids']
                    else:
                        current_turn_ids = self.tokenizer.apply_chat_template(
                            messages[:i + 1],
                            add_generation_prompt=False,
                            return_dict=True,
                            add_special_tokens=False,
                        )['input_ids']

                    # 计算当前这一轮消息（包括模板）引入的新 token 数量
                    new_tokens_count = len(current_turn_ids) - len_of_tokenized_so_far

                    if new_tokens_count <= 0:
                        continue

                    # 根据角色分配标签
                    if role == 'assistant':
                        generated_labels.extend([1] * new_tokens_count)
                    else:  # 'user', 'system'
                        generated_labels.extend([0] * new_tokens_count)

                    # 更新已处理的 token 总长度
                    len_of_tokenized_so_far = len(current_turn_ids)

                # 3、截断到最大长度
                input_ids = full_input_ids[:self.max_length]
                labels = generated_labels[:self.max_length]

                # 4、Attention Mask
                attention_mask = [1] * len(input_ids)

                if not (len(input_ids) == len(attention_mask) == len(labels)):
                    logger.error(f"第 {item} 项最终长度不匹配！(优化版)")
                    logger.error(f"  Input IDs len: {len(input_ids)}")
                    logger.error(f"  Attention Mask len: {len(attention_mask)}")
                    logger.error(f"  Labels len: {len(labels)}")
                    logger.warning(f"跳过第 {item} 项：因最终长度不匹配。")
                    raise ValueError(f"跳过第 {item} 项：因最终长度不匹配。")

                # Qwen-3 label 修正
                if self.is_qwen3:
                    self.fix_qwen3_labels(input_ids, labels)

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
