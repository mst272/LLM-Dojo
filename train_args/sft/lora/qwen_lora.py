from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments, SchedulerType, IntervalStrategy
from transformers.training_args import OptimizerNames


@dataclass
class TrainArgument(TrainingArguments):
    """
    训练参数, 直接在这里修改default即可
    """
    output_dir: str = field(default='./out', metadata={"help": "模型训练完成后的保存路径"})
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮次"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "训练的batch size"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "是否使用梯度累计"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累计的步长"})
    learning_rate: float = field(default=2e-4, metadata={"help": "学习率"})
    logging_steps: int = field(default=500, metadata={"help": "打印的步长"})
    save_steps: int = field(default=500, metadata={"help": "多少步长保存一次"})
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "If a value is passed, will limit the total "
                                                                         "amount of checkpoints. Deletes the older "
                                                                         "checkpoints in"})
    lr_scheduler_type: Union[SchedulerType, str] = field(default="constant_with_warmup",
                                                         metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=100, metadata={"help": "Linear warmup over warmup_steps."})
    optim: Union[OptimizerNames, str] = field(default='paged_adamw_32bit', metadata={"help": "The optimizer to use."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    fp16: bool = field(default=True, metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"})
    report_to: Optional[List[str]] = field(default='tensorboard', metadata={
        "help": "The list of integrations to report the results and logs to."})
    save_strategy: Union[IntervalStrategy, str] = field(default="steps",
                                                        metadata={"help": "The checkpoint save strategy to use."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={
        "help": "Remove columns not required by the model when using an nlp.Dataset."})
    bf16: bool = True
