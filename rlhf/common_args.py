from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TrainArgPath(Enum):
    PPO_ARGS = 'rlhf_args/ppo_config.py'
    RLOO_ARGS = 'rlhf_args/rloo_config.py'
    CPO_ARGS = 'rlhf_args/cpo_config.py'
    SimPO_ARGS = 'rlhf_args/simpo_config.py'
    CPOSimPO_ARGS = 'rlhf_args/cpo-simpo_config.py'


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    train_args_path: TrainArgPath = field(default=TrainArgPath.RLOO_ARGS.value,
                                          metadata={"help": "当前模式训练参数,目前支持 [PPO,RLOO,CPO,SimPO,CPOSimPO]"})
    # 微调方法相关选择与配置
    train_mode: str = field(default='lora', metadata={"help": "选择采用的训练方式：[qlora, lora, full]"})
    use_dora: bool = field(default=False,
                           metadata={"help": "仅在train_mode==lora时可以使用。是否使用Dora(一个基于Lora的变体)"})
    rlhf_type: str = field(default="RLOO",
                           metadata={"help": "选择使用的RLHF方法，目前支持[PPO,RLOO,CPO,SimPO,CPOSimPO]"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})

# max_len: int = field(default=1024, metadata={"help": "最大输入长度,dpo时该参数在dpo_config中设置"})
