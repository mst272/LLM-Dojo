from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluateArgs:
    """
    配置Evaluate的参数
    """
    max_new_tokens: int = 100
    max_length: int = 256
    do_sample: bool = False
    top_p: float = 0.95
    model_name_or_path: str = './'
