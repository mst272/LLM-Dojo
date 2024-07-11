import importlib
import os
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    BitsAndBytesConfig,
    Qwen2ForSequenceClassification,
    set_seed
)
import torch
import torch.nn as nn
from trl.trainer.rloo_trainer import RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from common_args import CommonArgs
import bitsandbytes as bnb


# parser = HfArgumentParser(CommonArgs, RLOOConfig)
# config = parser.parse_args_into_dataclasses()[0]


def load_config(args):
    # 根据config_option加载相应的配置
    module_path = args.train_args_path.replace("/", ".").rstrip(".py")
    # 动态导入模块
    module = importlib.import_module(module_path)
    # 每个模块导入的类名均为TrainArgument
    class_name = args.rlhf_type + "Config"
    # 使用getattr获取模块中的类
    argument = getattr(module, class_name)
    train_argument = argument()
    return train_argument


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names


def main():
    parser = HfArgumentParser((CommonArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    # 根据CommonArgs中的config_option动态加载配置
    config = load_config(args)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

    model_kwargs = dict(
        trust_remote_code=True
    )

    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if config.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs.update(quantization_config=quantization_config)

    # 如果模型不支持AutoModelForSequenceClassification需要再对应config文件中添加映射
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1,
                                                                      **model_kwargs)

    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    if args.train_mode == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=find_all_linear_names(policy),
            r=args.lora_rank,  # Lora 秩
            lora_alpha=args.lora_alpha,  # Lora alpha，具体作用参见 Lora 原理
            lora_dropout=args.lora_dropout  # Dropout 比例
        )


if __name__ == "__main__":
    main()
