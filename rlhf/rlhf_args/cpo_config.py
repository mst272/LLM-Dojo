from dataclasses import dataclass
from transformers import HfArgumentParser
from trl import CPOConfig


@dataclass
class CPOConfig(CPOConfig):
    # TrainingArguments的相关参数

    eval_samples: int = 30
    """eval sample的数量，注意不能少于batchsize*gradient_accumulation_steps"""


