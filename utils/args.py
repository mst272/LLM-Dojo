from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum


class TrainMode(Enum):
    QLORA = 'qlora'
    LORA = 'lora'
    FULL = 'full'


class TrainArgPath(Enum):
    SFT_LORA_QLORA_BASE = 'train_args/sft/lora_qlora/base.py'
    DPO_LORA_QLORA_BASE = 'train_args/dpo/dpo_config.py'


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    # Deepspeed相关参数
    local_rank: int = field(default=1, metadata={"help": "deepspeed所需参数,单机无需修改"})

    train_args_path: TrainArgPath = field(default=TrainArgPath.SFT_LORA_QLORA_BASE.value,
                                          metadata={"help": "当前模式的训练参数,分为sft和dpo参数"})
    max_len: int = field(default=1024, metadata={"help": "最大输入长度,dpo时该参数在dpo_config中设置"})
    max_prompt_length: int = field(default=512, metadata={
        "help": "dpo时，prompt的最大长度，适用于dpo_single,dpo_multi时该参数在dpo_config中设置"})
    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})
    model_name_or_path: str = field(default='./', metadata={"help": "下载的所需模型路径"})

    # 微调方法相关选择与配置
    train_mode: TrainMode = field(default=TrainMode.LORA.value,
                                  metadata={"help": "选择采用的训练方式：[qlora, lora, full]"})
    use_dora: bool = field(default=False, metadata={"help": "仅在train_mode==lora时可以使用。是否使用Dora(一个基于lora的变体) "
                                                            "目前只支持linear and Conv2D layers."})

    task_type: str = field(default="sft",
                           metadata={"help": "预训练任务：[pretrain, sft, dpo_multi, dpo_single]，目前支持sft,dpo"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
