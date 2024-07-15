import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser
from trl import RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from reward_args.model_config import OurModelConfig
from reward_args.reward_config import RewardConfig
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


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
    parser = HfArgumentParser((RewardConfig, OurModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # 如果模型不支持AutoModelForSequenceClassification需要在对应config文件中添加映射
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path, num_labels=1, **model_kwargs
        )
    except Exception as e:
        assert False, "模型不支持AutoModelForSequenceClassification需要在对应config文件中添加映射"
    model_config.lora_target_modules = find_all_linear_names(model)
    ################
    # Dataset
    ################
    raw_datasets = pd.read_json(config.train_data_path, lines=True)
    for i in range(len(raw_datasets)):
        chosen = raw_datasets['chosen'][i]
        rejected = raw_datasets['rejected'][i]
        raw_datasets.loc[i, 'chosen'] = tokenizer.apply_chat_template(chosen, tokenize=False)
        raw_datasets.loc[i, 'rejected'] = tokenizer.apply_chat_template(rejected, tokenize=False)
    raw_datasets = Dataset.from_pandas(raw_datasets, preserve_index=False)
    # 设置训练与测试集
    datasets = raw_datasets.train_test_split(test_size=0.1)

    def preprocess(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # 数据处理操作
    datasets = datasets.map(
        preprocess,
        batched=True,
        num_proc=4,
    )
    # 长度截断
    if config.max_length is not None:
        datasets = datasets.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length and len(
                x["input_ids_rejected"]) <= config.max_length
        )
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
