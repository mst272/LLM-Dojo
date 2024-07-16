from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class CPOConfig(TrainingArguments):
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None

    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "hinge", "ipo", "simpo"] = "sigmoid"
    disable_dropout: bool = True
    cpo_alpha: float = 1.0
    simpo_gamma: float = 0.5

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None

    model_init_kwargs: Optional[Dict] = None

    dataset_num_proc: Optional[int] = None
