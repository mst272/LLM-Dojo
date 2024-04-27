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
    train_mode: str = field(default="qlora", metadata={"help": "选择采用的训练方式：[full, qlora, lora]"})
