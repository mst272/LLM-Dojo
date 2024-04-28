from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    max_len: int = field(metadata={"help": "最大输入长度"})
    train_data_path: Optional[str] = field(metadata={"help": "训练集路径"})
    model_name_or_path: str = field(metadata={"help": "下载的所需模型路径"})
    template_name: str = field(default="", metadata={"help": "sft时的数据格式"})
    train_mode: str = field(default="qlora", metadata={"help": "选择采用的训练方式：[qlora, lora]"})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[pretrain, sft, dpo]"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
