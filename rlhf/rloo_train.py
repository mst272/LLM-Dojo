from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Phi3ForSequenceClassification,
    HfArgumentParser,
    Qwen2ForSequenceClassification
)

from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

#from ppo_args.model_config import ModelConfig
from rloo_args.rloo_config import RLOOConfig


parser = HfArgumentParser(RLOOConfig)
config = parser.parse_args_into_dataclasses()[0]
# remove output_dir if exists
# shutil.rmtree(config.output_dir, ignore_errors=True)

################
# Model & Tokenizer
################
loraconfig = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

tokenizer = AutoTokenizer.from_pretrained(
    config.sft_model_path,
    padding_side="left",
    trust_remote_code=True,
)
# 需要根据不同模型的情况进行适配，或者直接采用下面的OPENAI的原始设置
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

reward_model = Phi3ForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)

ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, trust_remote_code=True)
policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, trust_remote_code=True)


ref_policy = get_peft_model(ref_policy, loraconfig)
policy = get_peft_model(policy, loraconfig)
################
# Dataset
################
raw_datasets = load_dataset(config.train_data_path, split="train")
eval_samples = 20
train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
dataset_text_field = "prompt"


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=4,  # multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )


################
# Training
################
trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
    )
trainer.train()
trainer.save_model(config.output_dir)
trainer.generate_completions()