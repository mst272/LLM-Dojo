import os
import stat
from loguru import logger
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import bitsandbytes as bnb



def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names

def create_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                              # llama不支持fast
                                              use_fast=False if config.model_type == 'llama' else True
                                              )

    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    assert tokenizer.bos_token_id is not None, "bos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')

    return tokenizer


def create_model(args, train_args):
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')
    # 确定训练的精度
    torch_dtype = torch.float16 if train_args.fp16 else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)


    if args.train_mode == 'qlora':
        # 基本的qlora可以直接在加载模型中设置参数，也可以通过BitsAndBytesConfig进行一些设置
        pass
    elif args.train_mode == 'lora':
        # 找到所有linear层
        target_modules = find_all_linear_names(model, args.train_mode)