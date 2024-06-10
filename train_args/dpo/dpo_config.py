"""
一般来说这里的参数是各个模型都通用的
"""
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal, Dict
from transformers import TrainingArguments, SchedulerType, IntervalStrategy
from transformers.training_args import OptimizerNames
from trl import DPOConfig


@dataclass
class TrainArgument(TrainingArguments):
    """
    训练参数, 直接在这里修改default即可
    """
    output_dir: str = field(default='', metadata={"help": "模型训练完成后的保存路径"})
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮次"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "训练的batch size"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "是否使用梯度累计"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累计的步长"})
    learning_rate: float = field(default=2e-4, metadata={"help": "学习率"})
    logging_steps: int = field(default=100, metadata={"help": "打印的步长"})
    save_steps: int = field(default=500, metadata={"help": "多少步长保存一次"})
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="no", metadata={"help": "The evaluation "
                                                                                              "strategy to use."}, )
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={"help": "The checkpoint save "
                                                                                           "strategy to use."}, )
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "If a value is passed, will limit the total "
                                                                         "amount of checkpoints. Deletes the older "
                                                                         "checkpoints in"})
    lr_scheduler_type: Union[SchedulerType, str] = field(default="constant_with_warmup",
                                                         metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=10, metadata={"help": "Linear warmup over warmup_steps."})
    optim: Union[OptimizerNames, str] = field(default='paged_adamw_32bit', metadata={"help": "The optimizer to use."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    report_to: Optional[List[str]] = field(default='tensorboard', metadata={
        "help": "The list of integrations to report the results and logs to."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={
        "help": "Remove columns not required by the model when using an nlp.Dataset."})
    bf16: bool = field(default=True, metadata={
        "help": ("Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                 " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
                 )
    })
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"})

    # Deepspeed训练相关参数，不使用时设置为default=None
    deepspeed: Optional[str] = field(default='./train_args/deepspeed_config/ds_config_zero2.json', metadata={"help": "启用Deepspeed时需要的config文件"})
# ---------------------------------------------------------------------------------------------------------------------
    # 上面参数是常规TrainingArguments设置，下面参数则是dpo配置参数。下面为DPOConfig默认配置。
    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal[
        "sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo_hard", "nca_pair", "robust"
    ] = "sigmoid"
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None


