import heapq
import time
import shutil
import itertools
from dataclasses import dataclass
from random import random
from typing import List, Optional
import re
from contextlib import contextmanager
from transformers.integrations import WandbCallback
import torch
from datasets import load_dataset
import wandb
from accelerate.utils import gather_object
import os
import deepspeed
from trl.models.utils import unwrap_model_for_generation
from accelerate import Accelerator
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from tqdm.auto import tqdm
from transformers import GenerationConfig, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_callback import ExportableState
import evaluate


def _generate_completions(
        prompts: list[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
        generation_config: Optional[GenerationConfig],
        batch_size: int = 1,
) -> list[str]:
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (list[str]): A list of input prompts for which completions are to be generated.
        model (PreTrainedModel): The pre-trained model to be used for generation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for encoding and decoding.
        accelerator (Accelerator): The accelerator to be used for model execution.
        generation_config (GenerationConfig): Configuration for text generation.
        batch_size (int, optional): The number of prompts to process in each batch. Default is 1.

    Returns:
        list[str]: A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        # 创建分布式安全的进度条（仅在主进程显示）
        total_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)

        progress_bar = tqdm(
            total=total_batches,
            desc="Generating Completions",
            disable=not accelerator.is_main_process,  # 非主进程禁用进度条
            dynamic_ncols=True  # 自动适应终端宽度
        )

        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx: idx + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            print("Input shape:", tokenized_batch.input_ids.shape)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations):
                # Remove prompt from generation
                generation = generation[len(prompt):]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)
            # 更新进度条（自动处理分布式同步）
            progress_bar.update(1)
        progress_bar.close()
    return completions
