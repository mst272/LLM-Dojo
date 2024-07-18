from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class CPOConfig(TrainingArguments):
    max_length: Optional[int] = None
    """The maximum length of the sequences in the batch."""
    max_prompt_length: Optional[int] = None
    """The maximum length of the prompt."""
    max_target_length: Optional[int] = None
    """The maximum length of the target."""

    beta: float = 0.1
    """The beta factor in CPO loss."""
    label_smoothing: float = 0
    """The label smoothing factor. This argument is required if you want to use the default data collator."""
    loss_type: Literal["sigmoid", "hinge", "ipo", "simpo"] = "sigmoid"
    """The type of loss to use."""
    disable_dropout: bool = True
    """Whether or not to disable dropouts in `model`."""
    cpo_alpha: float = 1.0
    """A hyperparameter that controls the strength of the BC regularizer in CPO training."""
    simpo_gamma: float = 0.5
    """A target reward margin for the SimPO loss, used only when the "simpo" option is enabled."""

    label_pad_token_id: int = -100
    """The label pad token id."""
    padding_value: int = None
    """The padding value if it is different to the tokenizer's pad_token_id."""
    truncation_mode: str = "keep_end"
    """The truncation mode to use, either `keep_end` or `keep_start`."""
    generate_during_eval: bool = False
    """Whether to sample and log generations during evaluation step."""
    is_encoder_decoder: Optional[bool] = None
    """If no model is provided, we need to know if the model_init returns an encoder-decoder."""
    model_init_kwargs: Optional[Dict] = None
    """Dict of Optional kwargs to pass when instantiating the model from a string"""

    dataset_num_proc: Optional[int] = None
    """The number of workers to use to tokenize the data. Defaults to None."""
