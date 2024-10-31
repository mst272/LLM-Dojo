from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class TrainArgument(TrainingArguments):
    """
    训练参数, 直接在这里修改即可
    """
    output_dir: str = field(default='', metadata={"help": "模型训练完成后的保存路径"})
    num_train_epochs: int = 1,

    per_device_train_batch_size: int = 2
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 16,

    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: int = 2
    lr_scheduler_type: str = "constant_with_warmup",
    warmup_steps: int = 10
    optim: str = 'adamw_torch'
    report_to: str = 'tensorboard'
    remove_unused_columns: bool = False
    bf16: bool = True
    fp16: bool = False

    # Deepspeed训练相关参数，不使用时设置为default=None
    deepspeed: Optional[str] = field(default=None, metadata={"help": "启用Deepspeed时需要的config文件"})
