from dataclasses import dataclass

torch_dtype = ['bf16', 'fp16', 'fp32']
task_name = ['humaneval']


@dataclass
class EvaluateArgs:
    """
    配置Evaluate的参数
    """

    """任务名，支持的任务见列表task_name"""
    task_name: str = 'humaneval'

    """模型生成的一些参数设置"""
    max_new_tokens: int = 100
    torch_dtype: str = 'fp16'
    do_sample: bool = False
    top_p: float = 0.95
    temperature: int = 1

    """模型路径"""
    model_name_or_path: str = './'
    """是否仅评测模式，若为False，则下面的evaluate_data_path不用填"""
    evaluate_only: bool = False
    """仅评测时的文件路径"""
    evaluate_data_path: str = ''
    """模型生成的输出路径"""
    output_path: str = './'
    """模型评测时输出的信息：pass或错误"""
    save_logs_path: str = './'
    """模型结果保存路径"""
    save_metrics_path: str = './'
    """评测所需要的测试集"""
    data_file: str = ''
