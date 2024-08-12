import os
from os.path import join
from loguru import logger
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, \
    BitsAndBytesConfig, HfArgumentParser, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, cast_mixed_precision_params
from train_args import dpo_TrainArgument, sft_TrainArgument
import bitsandbytes as bnb
from utils.template import template_dict
from utils.data_process import MultiRoundDataProcess, DpoDataset
from utils.data_collator import SftDataCollator
from utils.args import CommonArgs
from datasets import load_dataset
from trl import DPOTrainer


def initial_args():
    parser = HfArgumentParser((CommonArgs,))
    args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if args.train_args_path == "sft_args":
        parser_b = HfArgumentParser((sft_TrainArgument,))
        train_args, = parser_b.parse_args_into_dataclasses(args=remaining_args)
        print("Loaded instance sft_args")
    elif args.train_args_path == "dpo_args":
        parser_c = HfArgumentParser((dpo_TrainArgument,))
        train_args, = parser_c.parse_args_into_dataclasses(args=remaining_args)
        print(f"Loaded instance dpo_args")
    else:
        raise ValueError("Invalid train_args_path choice")

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
    assert train_mode in ['lora', 'qlora']
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
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.bos_token is None:  # qwen没有bos_token，要设置一下，不然dpo train时会报错。
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id

    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')

    return tokenizer


def create_model(args, train_args):
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')
    # 确定训练的精度
    torch_dtype = torch.bfloat16 if train_args.bf16 else torch.float32
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if train_args.gradient_checkpointing else True,  # The cache is only used for generation,
        # fix bug
        # device_map='auto'
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
        if args.task_type in ['pretrain', 'sft']:  # 如果是dpo的话就不执行
            # QLoRA: casts all the non int8 modules to full precision (fp32) for stability
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    elif args.train_mode == 'lora':
        model = load_model(model_kwargs)
        if hasattr(model, 'enable_input_require_grads'):
            # 不加可能报错
            model.enable_input_require_grads()
    elif args.train_mode == 'full':
        model = load_model(model_kwargs)

    if args.train_mode == 'full':
        peft_config = None
    else:
        # peft_config配置
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            use_dora=args.use_dora
        )

    # peft_model 配置
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        if not train_args.bf16:
            cast_mixed_precision_params(model, dtype=torch.float16)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'peft_config': peft_config,
    }


def load_sft_dataset(args, tokenizer):
    # if args.template_name not in template_dict.keys():
    #     raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    # template = template_dict[args.template_name]
    train_dataset = MultiRoundDataProcess(args.train_data_path, tokenizer, args.max_len)
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    # 官方dpo方法，一般情况下推荐使用这个，而且按照数据格式进行处理，多轮也可改为单轮。
    if args.task_type == 'dpo_multi':
        if tokenizer.chat_template is None:
            tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
        train_dataset = load_dataset(data_files=args.train_data_path, path='json')

        def process(row):
            row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
            row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
            return row

        train_dataset = train_dataset.map(process)
        train_dataset = train_dataset['train']
        return train_dataset
    # 使用自己构建的dpo dataset，用于自己科研或魔改使用。
    elif args.task_type == 'dpo_single':
        template = template_dict['qwen']
        train_dataset = DpoDataset(args.train_data_path, tokenizer, args.max_len, args.max_prompt_length, template)
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
    elif args.task_type == 'dpo_multi' or args.task_type == 'dpo_single':
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None

    # sft or pretrain
    if args.task_type == 'sft' or args.task_type == 'pretrain':
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
    else:
        trainer = DPOTrainer(
            model,
            ref_model=None,
            args=train_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config
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
