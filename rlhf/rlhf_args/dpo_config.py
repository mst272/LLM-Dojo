from dataclasses import dataclass
from rlhf_args.base_config import BaseConfig
from typing import Literal
from trl import DPOConfig as TrlDPOConfig


@dataclass
class DPOConfig(BaseConfig, TrlDPOConfig):
    """
    训练参数, 可直接在此修改. 想看更多参数可直接在TrlDPOConfig中去看
    """
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: Literal[
        "sigmoid",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_pair",
        "apo_zero",
        "apo_down",
    ] = "sigmoid"
    label_pad_token_id: int = -100
