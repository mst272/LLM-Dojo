from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    train_data_path: str = field(default='', metadata={"help": "训练数据路径"})
    # 微调方法相关选择与配置
    rlhf_type: str = field(default="DPO",
                           metadata={"help": "选择使用的RLHF方法，目前支持[PPO,RLOO,DPO,CPO,SimPO,CPOSimPO,Reward]"})
    train_mode: str = field(default='lora', metadata={"help": "选择采用的训练方式：[qlora, lora, full]"})

    # model qlora lora相关配置
    model_name_or_path: str = './'
    use_dora: bool = field(default=False, metadata={"help": "仅在train_mode==lora时可以使用。是否使用Dora(一个基于Lora的变体)"})
    lora_rank: Optional[int] = field(default=32, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
