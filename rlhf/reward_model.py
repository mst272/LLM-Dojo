import warnings
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from args.model_config import Model_Config
from args.reward_config import RewardConfig

parser = HfArgumentParser((RewardConfig, Model_Config))
config, model_config = parser.parse_args_into_dataclasses()
config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

################
# Model & Tokenizer
################
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name_or_path, num_labels=1, **model_kwargs
)

if model_config.lora_task_type != "SEQ_CLS":
    warnings.warn(
        "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
        " Make sure to pass --lora_task_type SEQ_CLS when using this script."
    )

################
# Dataset
################
raw_datasets = load_dataset("Anthropic/hh-rlhf")
# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets

def preprocess_function(examples):
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

# Preprocess the dataset and filter out examples that are longer than args.max_length
raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
raw_datasets = raw_datasets.filter(
    lambda x: len(x["input_ids_chosen"]) <= config.max_length and len(x["input_ids_rejected"]) <= config.max_length
)
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

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
trainer.push_to_hub()
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
print(metrics)