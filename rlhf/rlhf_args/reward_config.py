from dataclasses import dataclass
from base_config import BaseConfig
from trl import RewardConfig as TrlRewardConfig


@dataclass
class RewardConfig(BaseConfig, TrlRewardConfig):
    pass
