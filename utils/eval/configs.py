from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from transformers import TrainingArguments


@dataclass
class GenerationConfig:
    """
    generation config, Vllm or model.generate

    Parameters:
        max_new_tokens (`int`, *optional*, defaults to `16`):
            Maximum number of tokens to generate for each prompt
        num_generation (`int`, *optional*, defaults to `1`):
            Number of completions to generate for each prompt.
        temperature:
            generate temperature
        top_p:
        top_k:
        do_sample:
        min_p

    """
    num_generation: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_new_tokens: int = 1024
    do_sample: bool = False

    # unwrap_model_for_generation
    gather_deepspeed3_params: bool = True  # if OOM, False it.


@dataclass
class EvaluationConfig:
    """
    Common test config

    Parameters:
        num_samples: 在测试集中随机选数量
        freq:
        metrics:
    """
    # 基础评估设置
    num_samples: Optional[int] = None
    freq: int = 5
    metrics: str = 'code'  # ['code', 'em'] or metrics=[{'name': 'code',
    # 'weight': 0.7},{'name': 'em', 'weight': 0.3}]

    higher_better: bool = True
    prompts_apply_chat: bool = False

    use_vllm: bool = True  # whether to use vllm to generate, if false, use unwrap_model_for_generation
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8080
    vllm_server_timeout: float = 120.0

    per_device_test_batch_size: int = 1  # Only use when use_vllm is False

    # Checkpoint管理
    save_best_checkpoints: bool = True
    start_update_best_checkpoints: int = 20
    max_checkpoints: int = 3


if __name__ == "__main__":
    gen = GenerationConfig()
    print(gen.stop_strings)
