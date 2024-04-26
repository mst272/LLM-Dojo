from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    max_len: int = field(metadata={"help": "最大输入长度"})
    train_data_path: Optional[str] = field(metadata={"help": "训练集路径"}
                                           )
