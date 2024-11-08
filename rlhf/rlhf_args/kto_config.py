from dataclasses import dataclass
from rlhf_args.base_config import BaseConfig
from trl import KTOConfig as TrlKTOConfig


@dataclass
class KTOConfig(BaseConfig, TrlKTOConfig):
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
