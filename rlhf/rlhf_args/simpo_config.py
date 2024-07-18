from dataclasses import dataclass
from typing import Literal
from cpo_config import CPOConfig


@dataclass
class SimPOConfig(CPOConfig):
    """
    基于CPOConfig，只需修改
    """
    loss_type: Literal["sigmoid", "hinge", "ipo", "simpo"] = "simpo"
    """The type of loss to use."""
    cpo_alpha: float = 0
    """A hyperparameter that controls the strength of the BC regularizer in CPO training."""
    simpo_gamma: float = 0.5
    """A target reward margin for the SimPO loss, used only when the "simpo" option is enabled."""
