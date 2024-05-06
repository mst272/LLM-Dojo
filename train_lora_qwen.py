import os
import stat

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from loguru import logger
from utils.data_process import QwenDataProcess
from utils.data_collator import MyDataCollator


#  loraConfig
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # 这个不同的模型需要设置不同的参数，需要看模型中的attention层
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)

# 配置训练参数
outputfile = './output'
os.chmod(outputfile, stat.S_IRWXU)
args = TrainingArguments(
    output_dir=outputfile,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=2,
    gradient_checkpointing=True,
    save_steps=4,
    learning_rate=1e-4,
    save_on_each_node=True
)

# 加载 tokenizer
file = '../download_llm/qwen/Qwen-1_8B-Chat'
tokenizer = AutoTokenizer.from_pretrained(file, use_fast=False, trust_remote_code=True)

# QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
if tokenizer.__class__.__name__ == 'QWenTokenizer':
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.bos_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id

# 数据
data_file = './test.jsonl'
train_dataset = QwenDataProcess(data_file, tokenizer, 512)
data_collator = MyDataCollator(tokenizer, 512)

# 初始化model
model = AutoModelForCausalLM.from_pretrained(
    file,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
# 不加会报错
model.enable_input_require_grads()
# 加载lora参数
model = get_peft_model(model, config)
# 初始化Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# 计算模型参数量
total = sum(p.numel() for p in model.parameters())
logger.info("Total model params: %.2fM" % (total / 1e6))

# 开始训练
logger.info("*** starting training ***")
train_result = trainer.train()
# 保存最好的checkpoint
# final_save_path = join("./output/Qwen")
# trainer.save_model(final_save_path)  # Saves the tokenizer too
# 保存训练指标
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
