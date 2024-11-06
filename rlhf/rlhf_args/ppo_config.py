from dataclasses import dataclass
from trl import PPOConfig as TrlPPOConfig
from rlhf_args.base_config import BaseConfig


@dataclass
class PPOConfig(BaseConfig, TrlPPOConfig):
    eval_samples = 30 # eval数量
