import os
import pandas as pd
import json
from torch.utils.data import Dataset
from loguru import logger
from pathlib import Path
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

    def __init__(self, file_or_dir, tokenizer: PreTrainedTokenizerBase, max_length: int, train_args, auto_adapt=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = []  # 初始化数据列表
        self.auto_adapt = auto_adapt
        # —— 新增：识别是否 Qwen-3 —— #
        self.is_qwen3 = self.is_qwen3_tokenizer(tokenizer)
        # self.is_qwen3 = False

        self.train_args = train_args
        self.error_data = []

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
                        # logger.info(f"读取 {file_path} 文件")
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
            # logger.info(f"读取 {file_path} 文件")
        except Exception as e:
            logger.error(f"[Parquet] 读取文件 {file_path} 失败: {e}")
            raise

    def _read_directory(self, dir_path: str):
        """
        递归遍历目录，读取所有 .jsonl / .parquet
        """
        base = Path(dir_path)
        files = [str(p) for p in base.rglob("*") if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS]
        files.sort()  # 稳定顺序

        if not files:
            raise ValueError(f"目录 '{dir_path}' 中未找到 .jsonl / .parquet 文件")

        if self.train_args.local_rank == 0:
            logger.info(f"将在目录 '{dir_path}' 中读取 {len(files)} 个文件（递归）")

        for path in files:
            try:
                self._dispatch_single(path)
            except Exception as e:
                logger.error(f"跳过文件 {path}，原因: {e}")

    @staticmethod
    def is_qwen3_tokenizer(tokenizer) -> bool:
        """根据tokenizer.__class__.__name__词表大小判断是否 Qwen-3的tokenizer"""
        name = tokenizer.__class__.__name__
        vocab_size = len(tokenizer.get_vocab())
        tokenizer_model_max_length = tokenizer.model_max_length
        if ('Qwen2TokenizerFast' == name) and (vocab_size == 151669) and (tokenizer_model_max_length == 1048576):
            return 'qwen3coder'
        elif ('Qwen2TokenizerFast' == name) and (vocab_size == 151669):
            return 'qwen3'
        return 'Qwen2TokenizerFast' == name and vocab_size == 151669

    @staticmethod
    def fix_qwen3_labels(input_ids: List[int], labels: List[int]) -> None:
        """若为 Qwen-3，将特殊 pattern 的 labels 置 0（就地修改）"""
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

                # ========== ✨ 新增：Qwen3 的 多轮think 处理准备 ==========
                # 记录“最后一个 assistant”的下标；若不存在则为 -1
                last_asst_idx = -1

                for idx, m in enumerate(messages):
                    if m.get("role") == "assistant":
                        last_asst_idx = idx
                if self.is_qwen3 and self.is_qwen3 == 'qwen3':
                    # 计算模板插入的 think 块的 token 数
                    THINK_TOKS = 4 if last_asst_idx > 2 else 0
                else:
                    THINK_TOKS = 0
                # ====================================================

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

                    # ========== ✨ 新增：对“非最后一个 assistant”扣掉 think ==========
                    current_len = len(current_turn_ids)
                    if self.is_qwen3 and role == 'assistant' and i != last_asst_idx and THINK_TOKS > 0:
                        # 对于切片 messages[:i+1]，模板会把当前 assistant 当作“最后一个”而插入 think，
                        # 但在完整模板中它不是最后一个，需要扣掉这段多余的 token。
                        current_len -= THINK_TOKS
                    # ===============================================================

                    # 计算当前这一轮消息（包括模板）引入的新 token 数量
                    new_tokens_count = current_len - len_of_tokenized_so_far

                    if new_tokens_count <= 0:
                        continue

                    # 根据角色分配标签
                    if role == 'assistant':
                        if self.is_qwen3 == 'qwen3':  # 控制qwen3多轮训练只学最后一个assistant
                            bit = 1 if i == last_asst_idx else 0
                            generated_labels.extend([bit] * new_tokens_count)
                        else:
                            generated_labels.extend([1] * new_tokens_count)
                    else:  # 'user', 'system'
                        generated_labels.extend([0] * new_tokens_count)

                    # 更新已处理的 token 总长度
                    len_of_tokenized_so_far = current_len

                # 3、对于大于最大长度的数据进行剔除
                if len(full_input_ids) > self.max_length:
                    logger.error(f"大于最大长度，剔除数据")
                    return {
                        "input_ids": None,
                        "attention_mask": None,
                        "target_mask": None,
                    }
                else:
                    input_ids = full_input_ids[:self.max_length]
                    labels = generated_labels[:self.max_length]

                # 4、Attention Mask
                attention_mask = [1] * len(input_ids)

                if not (len(input_ids) == len(attention_mask) == len(labels)):
                    self.error_data.append
                    logger.error("")
                    logger.error(f"第 {item} 项最终长度不匹配！")
                    logger.error(f"  Input IDs len: {len(input_ids)}")
                    logger.error(f"  Attention Mask len: {len(attention_mask)}")
                    logger.error(f"  Labels len: {len(labels)}")
                    logger.warning(f"跳过第 {item} 项：因最终长度不匹配。数据: {data}。")
                    logger.error("")
                    # logger.warning(f"跳过第 {item} 项：因最终长度不匹配。数据: {data}。")
                    # raise ValueError(f"跳过第 {item} 项：因最终长度不匹配。")
                    return {
                        "input_ids": None,
                        "attention_mask": None,
                        "target_mask": None,
                    }

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
            # logger.exception(f"处理第 {item} 项时发生意外错误。 数据: {data}。")
            logger.warning(f"处理第 {item} 项时发生意外错误。 数据: {data}。")
            # 根据策略返回 None 或重新抛出异常
            inputs = {
                "input_ids": None,
                "attention_mask": None,
                "target_mask": None
            }  # 返回 None 以允许训练在过滤后继续
            return inputs
