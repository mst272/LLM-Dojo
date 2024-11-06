from dataclasses import dataclass
from typing import Optional
from rlhf_args.base_config import BaseConfig
from trl import RLOOConfig as TrlPLOOConfig


# 支持直接通过total_episodes确定训练步数，也支持通过在TrainingArguments中配置num_train_epochs确定训练步数。
@dataclass
class RLOOConfig(BaseConfig, TrlPLOOConfig):
    reward_model_path: str = "./"
    sft_model_path: str = "./"
    total_episodes: Optional[int] = None
    eval_samples = 30  # eval数量
