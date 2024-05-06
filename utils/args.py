from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum


# 目前支持的template 类型
class TemplateName(Enum):
    QWEN = 'qwen'
    YI = 'yi'


class TrainMode(Enum):
    QLORA = 'qlora'
    LORA = 'lora'


class TrainArgPath(Enum):
    SFT_LORA_QLORA_QWEN = 'train_args/sft/lora_qlora/qwen_lora.py'


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    train_args_path: TrainArgPath = field(default=TrainArgPath.SFT_LORA_QLORA_QWEN, metadata={"help": "当前模式的训练参数"})
    max_len: int = field(default=1024, metadata={"help": "最大输入长度"})
    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})
    model_name_or_path: str = field(default='./', metadata={"help": "下载的所需模型路径"})
    template_name: TemplateName = field(default=TemplateName.QWEN, metadata={"help": "sft时的数据格式"})

    # 微调方法相关选择与配置
    train_mode: TrainMode = field(default=TrainMode.LORA, metadata={"help": "选择采用的训练方式：[qlora, lora]"})
    use_dora: bool = field(default=False, metadata={"help": "仅在train_mode==lora时可以使用。是否使用Dora(一个基于lora的变体) "
                                                            "目前只支持linear and Conv2D layers."})

    task_type: str = field(default="sft", metadata={"help": "预训练任务：[pretrain, sft, dpo]"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
