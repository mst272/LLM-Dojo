import os
from os.path import join
import random
from typing import Optional

from loguru import logger
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, \
    BitsAndBytesConfig, HfArgumentParser, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, cast_mixed_precision_params
from train_args import sft_TrainArgument
import bitsandbytes as bnb
from utils.data_process import MultiRoundDataProcess
from utils.data_collator import SftDataCollator
from train_args.common_args import CommonArgs
from utils.eval.configs import EvaluationConfig, GenerationConfig
from utils.eval.callback import EvaluationCallback
from utils.eval.eval_metric import create_metric

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def initial_args():
    parser = HfArgumentParser((CommonArgs,))
    args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if args.train_args_path == "sft_args":
        if args.use_eval_in_train:
            parser_b = HfArgumentParser((sft_TrainArgument, EvaluationConfig, GenerationConfig))
            train_args, eval_args, gen_config = parser_b.parse_args_into_dataclasses(args=remaining_args)
        else:
            parser_b = HfArgumentParser((sft_TrainArgument,))
            train_args, = parser_b.parse_args_into_dataclasses(args=remaining_args)
    else:
        raise ValueError("Invalid train_args_path choice")

    if not os.path.exists(train_args.output_dir):
        os.makedirs(train_args.output_dir, exist_ok=True)
    set_seed(train_args.seed)

    assert sum([train_args.fp16, train_args.bf16]) == 1, "only one of fp16 and bf16 can be True"
    if args.use_eval_in_train:
        return args, train_args, eval_args, gen_config
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

    return tokenizer


def create_model(args, train_args):
    target_modules = None
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

        # logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        # model.print_trainable_parameters()

    return {
        'model': model,
        'peft_config': peft_config,
        'target_modules': target_modules
    }


def load_sft_dataset(args, tokenizer):
    train_dataset = MultiRoundDataProcess(args.train_data_path, tokenizer, args.max_len, args.auto_adapt)
    return train_dataset


def create_trainer(args, train_args, eval_args: Optional[EvaluationConfig] = None, gen_config: Optional[GenerationConfig] = None):
    """"
    Create Trainer，支持可选的评估功能
    Args:
        args: 通用参数
        train_args: 训练相关参数
        eval_args: 评估相关参数（可选）
        gen_config: 评估相关参数（可选）
    """
    # 1.Basic component initialization
    tokenizer = create_tokenizer(args)
    model_dict = create_model(args, train_args)
    model = model_dict['model']
    # peft_config = model_dict['peft_config']

    # 2. dataset process
    if args.task_type == 'sft':
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SftDataCollator(tokenizer, args.max_len)
    elif args.task_type == 'pretrain':
        pass

    # 3. log configuration
    log_out(args, train_args, tokenizer, train_dataset, model, model_dict['target_modules'], eval_args, gen_config)

    # 4. sft or pretrain
    if args.task_type == 'sft':
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
    elif args.task_type == 'pretrain':
        pass
    # 5. Add evaluation callbacks if eval_args is provided
    if eval_args is not None:
        test_datasets = load_dataset(
            path="json",
            data_files=args.test_datasets_path
        )['train']

        # 创建评估回调
        metrics = create_metric(eval_args)
        eval_callback = EvaluationCallback(
            trainer=trainer,
            test_datasets=test_datasets,
            generation_config=gen_config,
            num_samples=eval_args.num_samples,
            freq=eval_args.freq,
            metrics=metrics,
            max_checkpoints=eval_args.max_checkpoints,
            per_device_test_batch_size=eval_args.per_device_test_batch_size,
            higher_better=eval_args.higher_better,
            start_update_best_checkpoints=eval_args.start_update_best_checkpoints,
            use_vllm=eval_args.use_vllm,
            gather_deepspeed3_params=gen_config.gather_deepspeed3_params,
            prompts_apply_chat=eval_args.prompts_apply_chat,
            vllm_server_host=eval_args.vllm_server_host,
            vllm_server_port=eval_args.vllm_server_port,
            vllm_server_timeout=eval_args.vllm_server_timeout
        )
        trainer.add_callback(eval_callback)

    return trainer


def log_out(args, train_args, tokenizer, train_dataset, model, target_modules, eval_args, gen_config):
    total = sum(p.numel() for p in model.parameters())
    logger.add(join(train_args.output_dir, 'train.log'))
    if train_args.local_rank == 0:
        logger.info("train_args:{}".format(train_args))
        logger.info("common_args:{}".format(args))
        logger.info("\neval_args:{}".format(eval_args))
        logger.info("\ngen_config:{}".format(gen_config))
        logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
        logger.info(f'Loading model from base model: {args.model_name_or_path}')
        logger.info("Total model params: %.2fM" % (total / 1e6))
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        if args.train_mode != 'full':
            trainable_params, all_param = model.get_nb_trainable_parameters()
            logger.info(
                f"trainable params: {trainable_params:,d} || "
                f"all params: {all_param:,d} || "
                f"trainable%: {100 * trainable_params / all_param:.4f}"
            )
        logger.info(f'Train model with {args.task_type} task')
        logger.info(f'Train model with {args.train_mode}')
        logger.info(f'LoRA target module names: {target_modules}')
        logger.info(f'Loading data: {args.train_data_path}')
        logger.info(f"Training dataset samples:{len(train_dataset)}")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['target_mask']}.")
            logger.info(
                f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")


def main():
    # args, train_args = initial_args()
    # # 加载trainer
    # trainer = create_trainer(args, train_args)
    result = initial_args()
    if len(result) == 4:
        args, train_args, eval_args, gen_config = result
        # 加载trainer，需要传入eval_args
        trainer = create_trainer(args, train_args, eval_args, gen_config)
    else:
        args, train_args = result
        # 原有的trainer创建方式
        trainer = create_trainer(args, train_args)
    # 开始训练
    if train_args.local_rank == 0:
        logger.info("*** starting training ***")
    train_result = trainer.train()
    # Transformers 更新了自动保存最后训练结果
    # final_save_path = join(train_args.output_dir)
    # trainer.save_model(final_save_path)

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
