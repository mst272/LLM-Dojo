from .dpo.dpo_config import TrainArgument as dpo_TrainArgument
from .sft.lora_qlora.base import TrainArgument as sft_TrainArgument

__all__ = [
    "dpo_TrainArgument",
    "sft_TrainArgument",
]