from dataclasses import dataclass
from typing import Literal
from cpo_config import CPOConfig


@dataclass
class CPOSimPOConfig(CPOConfig):
    """
    基于CPOConfig，只需修改
    """
    loss_type: Literal["sigmoid", "hinge", "ipo", "simpo"] = "simpo"
    """The type of loss to use."""
    cpo_alpha: float = 0.5
    """combined use of CPO and SimPO, which enables more stable training and improved performance.A non-zero 
    cpo_alpha"""

