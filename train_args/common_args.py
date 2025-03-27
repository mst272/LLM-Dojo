from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TrainMode(Enum):
    QLORA = 'qlora'
    LORA = 'lora'
    FULL = 'full'


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    # Deepspeed相关参数，如出现报错可注释掉
    # local_rank: int = field(default=1, metadata={"help": "deepspeed所需参数,单机无需修改，如出现报错可注释掉或添加"})

    train_args_path: str = 'sft_args'  # 训练参数 默认sft_args
    max_len: int = field(default=1024, metadata={"help": "最大输入长度"})
    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})
    model_name_or_path: str = field(default='./', metadata={"help": "下载的所需模型路径"})

    # 训练方法相关选择与配置
    task_type: str = field(default="sft", metadata={"help": "预训练任务：目前支持sft"})
    train_mode: TrainMode = field(default='lora', metadata={"help": "选择采用的训练方式：[qlora, lora, full]"})
    use_dora: bool = field(default=False,
                           metadata={"help": "在train_mode==lora时可以使用。是否使用Dora(一个基于lora的变体)"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})

    # 是否自动适配template
    auto_adapt: bool = field(default=True, metadata={"help": "选择是否自动适配template，若为False,则直接使用输入数据"})
    # 是否训练中评测
    use_eval_in_train: bool = False
    test_datasets_path: Optional[str] = None
