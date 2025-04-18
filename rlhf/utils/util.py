import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType


class ArgumentParserPlus(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # noqa adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Finds the index of the first `True` value in each row of a boolean tensor. If no `True` value exists in a row,
    it returns the length of the row.

    Args:
        bools (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length), where `True` values indicate
                              the positions of interest.
        dtype (torch.dtype): The data type to use for the output indices (default is torch.long).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the index of the first `True` value in each row.
                      If a row has no `True` value, the index will be the length of the row.
    """

    # Get the length of each row (i.e., the number of columns in the last dimension)
    # row_len is a scalar representing the length of each sequence (sequence_length)
    row_len = bools.size(-1)

    # Calculate the index positions for the first `True` in each row
    # ~bools: Invert the boolean values (True becomes False and vice versa)
    # ~bools.type(dtype): Convert the inverted boolean tensor to the specified dtype (0 for True, 1 for False)
    # row_len * (~bools).type(dtype): For `False` values, this will give `row_len`, for `True` values it gives 0.
    # torch.arange(row_len, dtype=dtype, device=bools.device): Generates a tensor with values [0, 1, 2, ..., row_len-1]
    # for each row. Shape: (sequence_length,)
    # zero_or_index: Shape (batch_size, sequence_length). This tensor contains the indices for `True` values and `row_len`
    # for `False` values.
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)

    # Return the minimum value in each row (i.e., the first `True` index or `row_len` if none exist)
    # torch.min(zero_or_index, dim=-1).values: This returns the minimum value in each row, which corresponds to the first
    # `True` value's index or `row_len` if there is no `True` in that row.
    # The returned tensor has shape (batch_size,)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(
        model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function computes reward scores for a batch of query responses based on a pre-trained reward model.

    Args:
        model (torch.nn.Module): The pre-trained reward model.
        query_responses (torch.Tensor): Tensor containing the tokenized responses for which to compute rewards.
            Shape: (batch_size, sequence_length)
        pad_token_id (int): The ID used for padding tokens in the tokenized sequences.
        context_length (int): The length of the prompt or context preceding the completions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - reward_logits: The logits output from the model for all tokens in the sequences.
              Shape: (batch_size, sequence_length)
            - final_scores: The final reward scores, one for each sequence, after adjusting for sequence lengths.
              Shape: (batch_size,)
            - sequence_lengths: The lengths of each sequence (excluding padding).
              Shape: (batch_size,)

    For example:
        query_responses = torch.tensor([
            [token0, token1, token2, token3, 0, 0],
            [token0, token1, token4, 0, 0, 0]
            ])  # 形状: (2, 6)

        attention_mask = query_responses != 0
            # [[1, 1, 1, 1, 0, 0],
            #  [1, 1, 1, 0, 0, 0]]

        position_ids = attention_mask.cumsum(1) - attention_mask.long()
            # [[0, 1, 2, 3, 4, 4],
            #  [0, 1, 2, 3, 3, 3]]

        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
            # 在此例中，填充 token 已为 0，无变化

        reward_logits = torch.tensor([
            [[r0_0], [r0_1], [r0_2], [r0_3], [r0_4], [r0_5]],
            [[r1_0], [r1_1], [r1_2], [r1_3], [r1_4], [r1_5]]
        ])  # 形状: (2, 6, 1)

        query_responses[:, 2:] == 0
            # [[False, False, True, True],
            #  [False, True, True, True]]

        sequence_lengths = first_true_indices(...) - 1 + 2
            # first_true_indices = [2, 1]
            # sequence_lengths = [2-1+2, 1-1+2] = [3, 2]

        final_scores = reward_logits[torch.arange(2), [3, 2]].squeeze(-1)
            # = reward_logits[[0,1], [3, 2]]  --->   reward_logits[0, 3, :],  reward_logits[1, 2, :]
            # = [r0_3, r1_2]，形状: (2,)
    """

    # Create an attention mask where tokens that are not padding have a value of 1, and padding tokens have a value of 0
    # Shape: (batch_size, sequence_length)
    attention_mask = query_responses != pad_token_id

    # Calculate position IDs for each token, considering the cumulative sum of the attention mask (to exclude padding)
    # Shape: (batch_size, sequence_length)
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum

    # Access the LM backbone from the reward model using its base model prefix
    lm_backbone = getattr(model, model.base_model_prefix)

    # Replace padding tokens with zeros in the input IDs (so padding tokens won't affect the model's processing)
    # Shape: (batch_size, sequence_length)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])  # (batch_size, sequence_length)

    # Calculate the length of each sequence by finding the first occurrence of a padding token after the context
    # sequence_lengths shape: (batch_size,)
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    assert (
            reward_logits.shape[-1] == 1
    ), "Reward model should output a single scalar per token. Check if you added `num_labels=1` when doing `AutoModelForSequenceClassification.from_pretrained(...)`."
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454

    # Return the reward logits for all tokens, the final reward scores for each sequence, and the sequence lengths
    return (
        # reward_logits shape: (batch_size, sequence_length)
        reward_logits,
        # final_scores shape: (batch_size,)
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(
            -1
        ),  # Shape: (batch_size,)
        sequence_lengths,
    )


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names


def is_right_apply_chat(tokenizer, prompt: List[Dict[str, str]], assistant_content: List[Dict[str, str]]) -> bool:
    """
    Checks if the assistant's content is correctly applied to the prompt in a chat template.
    Args:
        tokenizer: The tokenizer.
        prompt: The initial prompt message.
        assistant_content: The content provided by the assistant.
    Returns:
        bool: True if the assistant's content is correctly applied, False otherwise.
    """
    try:
        test_assistant = tokenizer.apply_chat_template(assistant_content, tokenize=False)
        test_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        conversation = copy.deepcopy(prompt)
        conversation.append(assistant_content[0])
        if tokenizer.apply_chat_template(conversation) == test_prompt + test_assistant:
            return True
        else:
            return False
    except Exception as e:
        return False


def fix_chat_template_if_needed(tokenizer, prompt: List[Dict[str, str]], chosen: List[Dict[str, str]],
                                rejected: List[Dict[str, str]]):
    """
    Fixes the chat template if needed.
    Args:
        tokenizer: The tokenizer.
        prompt: The initial prompt message.
        chosen: The chosen response, a list containing a single dictionary representing the chosen message.
        rejected: The rejected response, a list containing a single dictionary representing the rejected message.
        Returns:
        - tuple: A tuple containing the fixed prompt, fixed chosen response, and fixed rejected response.
    """
    conversation_chosen = copy.deepcopy(prompt)
    conversation_rejected = copy.deepcopy(prompt)
    conversation_chosen.append(chosen[0])
    conversation_rejected.append(rejected[0])
    conversation_chosen = tokenizer.apply_chat_template(conversation_chosen, tokenize=False)
    conversation_rejected = tokenizer.apply_chat_template(conversation_rejected, tokenize=False)
    # find position
    start_position = conversation_chosen.find(chosen[0]['content'][0])
    # The following is right
    fixed_prompt = conversation_chosen[:start_position]
    fixed_chosen = conversation_chosen[start_position:]
    fixed_rejected = conversation_rejected[start_position:]
    return fixed_prompt, fixed_chosen, fixed_rejected
