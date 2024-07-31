from dataclasses import dataclass, field
from typing import Dict, Literal, Optional,Union

from transformers.training_args import OptimizerNames
from trl import CPOConfig

@dataclass
class CPOConfig(CPOConfig):
    max_length: Optional[int] = None
    """The maximum length of the sequences in the batch."""
    max_prompt_length: Optional[int] = None
    """The maximum length of the prompt."""
    max_target_length: Optional[int] = None
    """The maximum length of the target."""

    beta: float = 0.1
    """The beta factor in CPO loss."""
    label_smoothing: float = 0
    """The label smoothing factor. This argument is required if you want to use the default data collator."""
    loss_type: Literal["sigmoid", "hinge", "ipo", "simpo"] = "sigmoid"
    """The type of loss to use."""
    disable_dropout: bool = True
    """Whether or not to disable dropouts in `model`."""
    cpo_alpha: float = 1.0
    """A hyperparameter that controls the strength of the BC regularizer in CPO training."""
    simpo_gamma: float = 0.5
    """A target reward margin for the SimPO loss, used only when the "simpo" option is enabled."""

    label_pad_token_id: int = -100
    """The label pad token id."""
    padding_value: int = None
    """The padding value if it is different to the tokenizer's pad_token_id."""
    truncation_mode: str = "keep_end"
    """The truncation mode to use, either `keep_end` or `keep_start`."""
    generate_during_eval: bool = False
    """Whether to sample and log generations during evaluation step."""
    is_encoder_decoder: Optional[bool] = None
    """If no model is provided, we need to know if the model_init returns an encoder-decoder."""
    model_init_kwargs: Optional[Dict] = None
    """Dict of Optional kwargs to pass when instantiating the model from a string"""

    dataset_num_proc: Optional[int] = None
    """The number of workers to use to tokenize the data. Defaults to None."""

    # TrainingArguments的相关参数
    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})
    output_dir: str = field(default='', metadata={"help": "模型训练完成后的保存路径"})
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮次，与num_ppo_epochs不同"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "训练的batch size"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "是否使用梯度累计"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累计的步长"})
    learning_rate: float = field(default=2e-4, metadata={"help": "学习率"})
    logging_steps: int = field(default=10, metadata={"help": "打印的步长"})
    save_steps: int = field(default=500, metadata={"help": "多少步长保存一次"})
    save_strategy: str = field(default="steps", metadata={"help": "save strategy"}, )
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "最大保存个数限制"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "scheduler type"})
    warmup_steps: int = field(default=10, metadata={"help": "Linear warmup over warmup_steps."})
    optim: Union[OptimizerNames, str] = field(default='adamw_torch', metadata={"help": "The optimizer to use."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    report_to: str = field(default='wandb', metadata={"help": "report the results and logs to."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={
        "help": "Remove columns not required by the model when using an nlp.Dataset."})
    bf16: bool = field(default=True, metadata={"help": "是否使用bf16精度"})
    fp16: bool = field(default=False, metadata={"help": "是否使用bf16精度"})
