import os
from os.path import join
from loguru import logger
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, \
    BitsAndBytesConfig, HfArgumentParser, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from utils.template import template_dict
from utils.data_process import CommonSingleRoundDataProcess
from utils.data_collator import SftDataCollator
from utils.args import CommonArgs
import importlib


def load_config(train_args_path):
    # 根据config_option加载相应的配置
    module_path = train_args_path.replace("/", ".").rstrip(".py")
    # 动态导入模块
    module = importlib.import_module(module_path)
    # 每个模块导入的类名均为TrainArgument
    class_name = "TrainArgument"

    # 使用getattr获取模块中的类
    TrainArgument = getattr(module, class_name)
    train_argument = TrainArgument()
    return train_argument


def initial_args():
    # parser = HfArgumentParser((CommonArgs, TrainArgument))
    # args, train_args = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser((CommonArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    # 根据CommonArgs中的config_option动态加载配置
    train_args = load_config(args.train_args_path)

    if not os.path.exists(train_args.output_dir):
        os.mkdir(train_args.output_dir)
    logger.add(join(train_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(train_args))
    logger.info("common_args:{}".format(train_args))
    set_seed(train_args.seed)

    assert sum([train_args.fp16, train_args.bf16]) == 1, "only one of fp16 and bf16 can be True"
    return args, train_args


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora_qlora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

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
    if tokenizer.__class__.__name__ == 'QWenTokenizer' or tokenizer.__class__.__name__ == 'Qwen2Tokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')

    return tokenizer


def create_model(args, train_args):
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')
    # 确定训练的精度
    torch_dtype = torch.float16 if train_args.fp16 else torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if train_args.gradient_checkpointing else True,  # The cache is only used for generation,
        # not for training.
        device_map='auto'
    )

    def load_model(model_kwargs):
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        return model

    if args.train_mode == 'qlora':
        # 基本的qlora可以直接在加载模型中设置参数，也可以通过BitsAndBytesConfig进行一些设置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
            bnb_4bit_compute_dtype=torch.float16 if train_args.fp16 else torch.bfloat16,  # 4位精度计算的数据类型。
            bnb_4bit_quant_type="nf4",  # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。
            bnb_4bit_use_double_quant=True  # 是否使用双精度量化。如果设置为True，则使用双精度量化。
        )
        model_kwargs.update(quantization_config=quantization_config)
        model = load_model(model_kwargs)
        # QLoRA: casts all the non int8 modules to full precision (fp32) for stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    elif args.train_mode == 'lora':
        # 是否使用dora
        model_kwargs.update(use_dora=args.use_dora)

        model = load_model(model_kwargs)
        if hasattr(model, 'enable_input_require_grads'):
            # 不加可能报错
            model.enable_input_require_grads()

    # peft_config配置
    target_modules = find_all_linear_names(model, args.train_mode)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )

    # peft_model 配置
    model = get_peft_model(model, peft_config)
    logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    model.print_trainable_parameters()

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'peft_config': peft_config
    }


def load_sft_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    logger.info('Loading data with CommonSingleRoundDataProcess')
    train_dataset = CommonSingleRoundDataProcess(args.train_data_path, tokenizer, args.max_len, template)
    return train_dataset


def create_trainer(args, train_args):
    tokenizer = create_tokenizer(args)
    model_dict = create_model(args, train_args)
    model = model_dict['model']
    peft_config = model_dict['peft_config']

    if args.task_type == 'sft':
        logger.info('Train model with sft task')
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SftDataCollator(tokenizer, args.max_len)
    elif args.task_type == 'pretrain':
        pass
    else:
        pass

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    return trainer


def main():
    args, train_args = initial_args()
    # 加载trainer
    trainer = create_trainer(args, train_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(train_args.output_dir)
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
