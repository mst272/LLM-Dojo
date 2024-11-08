import importlib
import os
from peft import LoraConfig, TaskType
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    BitsAndBytesConfig,
)
import torch
from accelerate import PartialState
from trl import DPOTrainer, CPOTrainer, PPOTrainer, RLOOTrainer, RewardTrainer, KTOTrainer
from common_args import CommonArgs
from utils.utils import find_all_linear_names
from loguru import logger
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WITH_REWARD_MODEL = ['RLOO', 'PPO']
USE_REF_MODEL = ['DPO', 'RLOO', 'KTO']

trainer_map = {
    'PPO': PPOTrainer,
    "RLOO": RLOOTrainer,
    "DPO": DPOTrainer,
    "CPO": CPOTrainer,
    "SimPO": CPOTrainer,
    "CPOSimPO": CPOTrainer,
    'Reward': RewardTrainer,
    "KTO": KTOTrainer
}

train_args_path = {
    'PPO': 'rlhf_args/ppo_config.py',
    "RLOO": 'rlhf_args/rloo_config.py',
    "DPO": 'rlhf_args/dpo_config.py',
    "CPO": 'rlhf_args/cpo_config.py',
    "SimPO": 'rlhf_args/simpo_config.py',
    "CPOSimPO": 'rlhf_args/cpo-simpo_config.py',
    'Reward': 'rlhf_args/reward_config.py',
    'KTO': 'rlhf_args/kto_config.py'
}


def load_config(args, remaining_args):
    # 根据config_option加载相应的配置
    module_path = train_args_path[args.rlhf_type].replace("/", ".").rstrip(".py")
    # 动态导入模块
    module = importlib.import_module(module_path)
    # 每个模块导入的类名均为TrainArgument
    class_name = args.rlhf_type + "Config"
    # 使用getattr获取模块中的类
    argument = getattr(module, class_name)

    parser_b = HfArgumentParser((argument,))
    train_args, = parser_b.parse_args_into_dataclasses(args=remaining_args)
    return train_args


def load_judge_reward():
    pass


def load_classification_reward(path, model_kwargs):
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1,
                                                                          **model_kwargs)
        return reward_model
    except Exception as e:
        assert False, "模型不支持AutoModelForSequenceClassification需要在对应config文件中添加映射"


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element['prompt'],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True
    )


def main():
    parser = HfArgumentParser((CommonArgs,))
    args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    # 根据CommonArgs中的config_option动态加载配置
    config = load_config(args, remaining_args)

    ################
    # Data
    ################
    train_dataset = load_dataset(data_files=args.train_data_path, path='json')

    ################
    # Model & Tokenizer
    ################
    tokenizer = load_tokenizer(args.model_name_or_path)

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.bfloat16,
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

    # 加载policy model
    if args.rlhf_type == 'Reward':
        policy = load_classification_reward(args.model_name_or_path, model_kwargs)
    else:
        policy = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.train_mode in ['lora', 'qlora']:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS if args.rlhf_type == 'Reward' else TaskType.CAUSAL_LM,
            target_modules=find_all_linear_names(policy),
            r=args.lora_rank,  # Lora 秩
            lora_alpha=args.lora_alpha,  # Lora alpha，具体作用参见 Lora 原理
            lora_dropout=args.lora_dropout,  # Dropout 比例
            use_dora=args.use_dora
        )
        ref_model = None  # if peft, the model with a disabled adapter
    elif args.rlhf_type in USE_REF_MODEL:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        lora_config = None

    # 决定是否加载Reward model
    if args.rlhf_type in WITH_REWARD_MODEL:
        # 如果模型不支持AutoModelForSequenceClassification需要在对应config文件中添加映射
        reward_model = load_classification_reward(config.reward_model_path, model_kwargs)

        # data process
        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        train_dataset = train_dataset.select(range(len(train_dataset) - config.eval_samples))
        eval_dataset = train_dataset.select(range(len(train_dataset) - config.eval_samples, len(train_dataset)))
        with PartialState().local_main_process_first():
            train_dataset = prepare_dataset(train_dataset, tokenizer)
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        if ref_model is None:
            ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                                             torch_dtype=torch.float16 if config.fp16 else torch.bfloat16)

    ################
    # Training
    ################
    trainer_kwargs_map = {
        "DPO": dict(
            model=policy,
            ref_model=ref_model,
            args=config,
            train_dataset=train_dataset['train'],
            eval_dataset=train_dataset['test'] if config.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        if args.rlhf_type in ['DPO', 'KTO']
        else dict()
        ,
        'CPO': dict(
            model=policy,
            args=config,
            train_dataset=train_dataset['train'],
            eval_dataset=train_dataset['test'] if config.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        if args.rlhf_type in ['CPO', 'SimPO', 'CPOSimPO']
        else dict()
        ,
        "PPO": dict(
        ),
        "RLOO": dict(
            config=config,
            processing_class=tokenizer,
            policy=policy,
            ref_policy=ref_model,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        if args.rlhf_type == 'RLOO'
        else dict()
        ,
        'Reward': dict(
            model=policy,
            processing_class=tokenizer,
            args=config,
            train_dataset=train_dataset['train'],
            eval_dataset=train_dataset['test'] if config.eval_strategy != "no" else None,
            peft_config=lora_config,
        )
        if args.rlhf_type == 'Reward'
        else dict()
    }

    trainer_kwargs_map['SimPO'] = trainer_kwargs_map['CPO'].copy()
    trainer_kwargs_map['CPOSimPO'] = trainer_kwargs_map['CPO'].copy()
    trainer_kwargs_map['KTO'] = trainer_kwargs_map['DPO'].copy()

    # 从字典中获取相应的 Trainer 类
    trainer_kwargs = trainer_kwargs_map.get(args.rlhf_type)
    TrainerClass = trainer_map.get(args.rlhf_type)
    if TrainerClass is None:
        raise ValueError(f"Unknown trainer type: {args.rlhf_type}")

    trainer = TrainerClass(**trainer_kwargs)
    trainer.train()
    # trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
