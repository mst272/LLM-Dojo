from dataclasses import dataclass, field
from typing import Optional

torch_dtype = ['bf16', 'fp16']
task_name = ['humaneval']


@dataclass
class EvaluateArgs:
    """
    配置Evaluate的参数
    """
    task_name: str = 'humaneval'
    evaluate_only: bool = False
    evaluate_data_path: str = ''
    max_new_tokens: int = 100
    torch_dtype: str = 'fp16'
    do_sample: bool = False
    top_p: float = 0.95
    temperature: int = 1
    model_name_or_path: str = './'
    output_path: str = './'
    data_file: str = ''
