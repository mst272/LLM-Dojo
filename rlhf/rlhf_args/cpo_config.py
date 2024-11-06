from dataclasses import dataclass
from rlhf_args.base_config import BaseConfig
from trl import CPOConfig as TrlCPOConfig


@dataclass
class CPOConfig(BaseConfig, TrlCPOConfig):
    pass


