import os
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal
from transformers import TrainingArguments, SchedulerType, IntervalStrategy
from transformers.training_args import OptimizerNames
from trl.trainer.utils import OnpolicyRuntimeConfig

@dataclass
class PPOConfig(OnpolicyRuntimeConfig, TrainingArguments):
    # common config
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    sanity_check: bool = False
    """wether to run in debug mode"""

    # batch size related config
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    num_sample_generations: int = 10
    """the number of debugging samples generations (i.e., `generate_completions` calls) throughout training"""

    # other config
    # base_model: str = "EleutherAI/pythia-160m"
    # """the name of the pretrained model to use"""
    response_length: int = 53
    """the length of the response"""
    stop_token: Optional[Literal["eos"]] = None
    """the stop token"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `stop_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    reward_model_path: str = "./"
    """the path to the reward model"""
    sft_model_path: str = "./"
    """the path to the sft model"""

    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""

    # TrainingArguments的相关参数
    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})
    output_dir: str = field(default='', metadata={"help": "模型训练完成后的保存路径"})
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮次"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "训练的batch size"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "是否使用梯度累计"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累计的步长"})
    learning_rate: float = field(default=2e-4, metadata={"help": "学习率"})
    logging_steps: int = field(default=100, metadata={"help": "打印的步长"})
    save_steps: int = field(default=500, metadata={"help": "多少步长保存一次"})
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={"help": "The checkpoint save "
                                                                                           "strategy to use."}, )
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "If a value is passed, will limit the total "
                                                                         "amount of checkpoints. Deletes the older "
                                                                         "checkpoints in"})
    lr_scheduler_type: Union[SchedulerType, str] = field(default="constant_with_warmup",
                                                         metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=10, metadata={"help": "Linear warmup over warmup_steps."})
    optim: Union[OptimizerNames, str] = field(default='adamw_torch', metadata={"help": "The optimizer to use."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    report_to: Optional[List[str]] = field(default='wandb', metadata={
        "help": "The list of integrations to report the results and logs to."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={
        "help": "Remove columns not required by the model when using an nlp.Dataset."})
    bf16: bool = field(default=True, metadata={"help": "是否使用bf16精度"})

    # Deepspeed训练相关参数，不使用时设置为default=None
    deepspeed: Optional[str] = field(default=None, metadata={"help": "启用Deepspeed时需要的config文件"})

    world_size: Optional[int] = 1
    """The number of processes (GPUs) to use"""
