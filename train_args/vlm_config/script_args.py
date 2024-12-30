from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArgs:
    """
    自定义参数
    """
    # Deepspeed相关参数，如出现报错可注释掉
    # local_rank: int = field(default=1, metadata={"help": "deepspeed所需参数,单机无需修改，如出现报错可注释掉或添加"})
    task_type: str = field(default='QA', metadata={"help": "任务类型，目前可选：[QA]"})
    '''多模态任务类型'''

    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})
    '''训练集路径'''

    train_mode: str = field(default='lora', metadata={"help": "选择对llm采用的训练方式：[qlora, lora, full]"})
    '''选择对llm采用的训练方式'''

    freeze_vision: bool = True
    '''训练是否冻结视觉层'''

    freeze_projector: bool = False
    '''训练是否冻结转接层'''

