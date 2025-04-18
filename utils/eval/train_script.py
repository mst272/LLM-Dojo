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
from eval_metric import CodeEvalMetric
from utils import MultiRoundDataProcess, SftDataCollator
from callback import EvaluationCallback

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 1
gradient_accumulation_steps = 2
num_train_epochs = 1

training_args = TrainingArguments(
    output_dir="./output/",
    report_to="wandb",  # this tells the Trainer to log the metrics to W&B
    per_device_train_batch_size=batch_size,
    bf16=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    num_train_epochs=num_train_epochs,
    # logging strategies
    logging_strategy="steps",
    logging_steps=2,
    torch_compile=False,
    remove_unused_columns=False,
    deepspeed='deepspeed_config/ds_config_zero3.json'
)

if __name__ == "__main__":
    model_name_or_path = '/Qwen2.5-Coder-32B-Instruct'
    train_data_path = 'train_data/fix_bash1k.jsonl'
    test_data_path = 'eval_train_test/test.jsonl'

    max_len = 4096
    auto_adapt = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    train_dataset = MultiRoundDataProcess(train_data_path, tokenizer, max_len, auto_adapt)

    test_dataset = load_dataset(path="json", data_files=test_data_path)
    test_dataset = test_dataset['train']

    data_collator = SftDataCollator(tokenizer, max_len)
    generate_config = GenerationConfig(
        max_new_tokens=4096,
        max_length=max_len,
        use_cache=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    # if os.environ.get('LOCAL_RANK', '0') == '0':  # 只在主进程中初始化
    #     wandb.init(project="huggingface")
    # wandb.init(project="huggingface")

    wandb_callback = EvaluationCallback(
        trainer=trainer,
        test_dataset=test_dataset,
        generation_config=generate_config,
        num_samples=6,
        freq=1,
        metric=CodeEvalMetric(),
        max_checkpoints=1,
        per_device_test_batch_size=1,
        higher_better=True,
        start_update_best_checkpoints=100
    )
    trainer.add_callback(wandb_callback)

    trainer.train()