from dataclasses import dataclass
from transformers import GenerationConfig


@dataclass
class GenerationConfig(GenerationConfig):
    max_new_tokens: int = 256
    max_length: int = 512
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False,
    use_cache: bool = False  # if gather_deepspeed3_params=False, the use_cache should be false


@dataclass
class EvalConfig:
    """评估系统配置"""
    # 基础评估设置
    enabled: bool = True
    evaluation_strategy: str = "steps"  # "steps" 或 "epoch"
    eval_steps: int = 100  # 如果 evaluation_strategy = "steps"
    eval_delay: int = 0  # 开始评估前等待的步
    use_vllm: bool = True  # 是否使用VLLM加速生成

    # 评估指标设置
    metrics: List[MetricConfig] = field(default_factory=lambda: [
        MetricConfig(
            name="code_eval",
            weight=1.0,
            config={"metric_path": "./metrics/code_eval"},
            higher_better=True
        )
    ])

    # Checkpoint管理
    save_best_checkpoints: bool = True
    max_checkpoints: int = 3
    metric_for_best_checkpoint: str = "aggregate_score"