from typing import Optional
import re

import numpy as np
from trl.models.utils import unwrap_model_for_generation
from accelerate import Accelerator
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig
)
from tqdm.auto import tqdm
import torch


def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def _generate_completions(
        prompts: list[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
        generation_config: Optional[GenerationConfig],
        batch_size: int = 1,
        gather_deepspeed3_params: bool = True,
) -> list[str]:
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (list[str]): A list of input prompts for which completions are to be generated.
        model (PreTrainedModel): The pre-trained model to be used for generation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for encoding and decoding.
        accelerator (Accelerator): The accelerator to be used for model execution.
        generation_config (GenerationConfig): Configuration for text generation.
        batch_size (int, optional): The number of prompts to process in each batch. Default is 1.
        gather_deepspeed3_params: bool = True: if OOM, False it.

    Returns:
        list[str]: A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    with unwrap_model_for_generation(model, accelerator,
                                     gather_deepspeed3_params=gather_deepspeed3_params) as unwrapped_model:
        # 创建分布式安全的进度条（仅在主进程显示）
        total_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)

        progress_bar = tqdm(
            total=total_batches,
            desc="Generating Completions",
            disable=not accelerator.is_main_process,  # 非主进程禁用进度条
            dynamic_ncols=True  # 自动适应终端宽度
        )

        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx: idx + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations):
                # Remove prompt from generation
                generation = generation[len(prompt):]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)
            # 更新进度条（自动处理分布式同步）
            progress_bar.update(1)
        progress_bar.close()
    return completions


def reason_post_process(code, index):
    """

    Args:
        code (str): 输入字符串。
        index (int/str): 当前字符串的序号 (索引)。

    Returns:
        str 或 int: 如果找到代码块，则返回代码块字符串；
                     否则，返回输入的字符串序号 (index)。
    """

    # Look for code blocks
    code_pattern = r'```(?:python|go|javascript|java|bash|js|cpp|cs|php)(.*?)```'
    code_match = re.findall(code_pattern, code, re.DOTALL)

    if code_match:
        # If code block exists, return its content (excluding the ``` markers)
        return code_match[-1].strip()
    else:
        # If no code block, return the solution content directly
        return str(index)
