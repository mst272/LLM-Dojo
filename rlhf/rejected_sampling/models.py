"""
Model implementations for different types of scoring models.
Contains classes and functions for classification models, API models, and local vLLM models.
"""
import asyncio
from typing import List, Dict, Tuple, Union
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)


# for classification model
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


# for classification model
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


class ClassificationModel:
    """Classification model for scoring using HuggingFace transformers."""

    def __init__(self, model_path: str, device: torch.device, max_batch_size: int):
        self.model_path = model_path
        self.device = device
        self.max_batch_size = max_batch_size
        self._setup_model()

    def _setup_model(self):
        """Initialize tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="right"
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(self.device).eval()

        # Initialize a data collator to handle dynamic padding of input sequences
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _batch_process(
            self,
            ds: Dataset,
            current_batch_size: int
    ) -> torch.Tensor:
        """Process dataset in batches with dynamic batch sizing."""
        # NOTE: two optimizations here:
        # 1. we sort by input_ids length to reduce padding at first
        # 1.1 note that this may cause slightly different results due to numerical issues.
        #   e.g., with sort: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1723242217
        #   e.g., without sort: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1723242476
        # 2. we shrink the batch size if we run out of memory (so initially we can use a large batch size)

        # Sort by length to optimize padding
        input_lengths = [len(x) for x in ds["input_ids"]]
        sorted_indices = np.argsort(input_lengths)
        scores = []
        i = 0

        while i < len(ds):
            with torch.no_grad():
                batch_data = ds[sorted_indices[i: i + current_batch_size]]
                try:
                    print(f"Processing: {i}:{i + current_batch_size}/{len(ds)}")
                    input_ids = self.data_collator(batch_data)["input_ids"].to(self.device)
                    _, score, _ = get_reward(
                        self.model,
                        input_ids,
                        self.tokenizer.pad_token_id,
                        0
                    )
                    # score = (batch_size, )
                    scores.extend(score.cpu().tolist())  # convert the tensor score to a list
                    i += current_batch_size
                except torch.cuda.OutOfMemoryError:
                    if current_batch_size == 1:
                        raise ValueError("Out of memory even with batch size 1")
                    current_batch_size //= 2
                    print(f"Reducing batch size to {current_batch_size}")
                    continue

        # Restore original order
        scores = np.array(scores)
        scores = scores[np.argsort(sorted_indices)]
        return torch.tensor(scores)

    def get_scores(self, shard: List[str], **kwargs) -> torch.Tensor:
        """
        Process a shard of data using the classification model.

        This function processes a shard (subset) of data using a specified model. It tokenizes the data,
        runs it through the model to get reward scores, and handles out-of-memory errors by adjusting the batch size.

        Args:
            shard (List[str]): A list of strings representing the shard of data to be processed.

        Returns:
            torch.Tensor: A tensor containing the reward scores for each item in the shard.
                          Shape: (num_items_in_shard,)
        """
        # Convert to dataset and tokenize
        raw_ds = Dataset.from_list(shard)
        # Apply a tokenization function to each item in the dataset
        ds = raw_ds.map(
            lambda x: {"input_ids": self.tokenizer.apply_chat_template(x["messages"])},
            remove_columns=raw_ds.column_names,
            num_proc=4
        )

        return self._batch_process(ds, self.max_batch_size)


class APIModel:
    """API-based model for scoring (e.g., GPT-3.5, GPT-4)."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def _generate_scores(self, data: List[Dict]) -> List[float]:
        """Generate scores using API calls."""
        # TODO: Implement actual API calls
        pass

    @staticmethod
    def format_conversation(messages: List[dict]) -> str:
        """Format conversation messages to string."""
        formatted = []
        for msg in messages:
            role = "User A" if msg["role"] == "user" else "User B"
            formatted.append(f"{role}: {msg['content'].strip()}")
        return "\n".join(formatted)

    def get_score(self, shard: List[str]) -> torch.Tensor:
        """
        Process a shard of data using the API model.

        shard (List[str]): A list of strings representing the shard of data to be processed.
        """
        raw_ds = Dataset.from_list(shard)
        ds = raw_ds.map(
            lambda x: {"prompt": self.format_conversation(x["messages"][:-1])},
            num_proc=4
        )

        prompts = ds["prompt"]
        responses = ds["model_completion"]

        data = [
            {"prompt": p, "response": r}
            for p, r in zip(prompts, responses)
        ]

        scores = asyncio.run(self._generate_scores(data))
        return torch.tensor(scores)


class VLLMModel:
    """Local vLLM model for scoring."""

    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        # TODO: Initialize vLLM model

    def process_shard(self, shard: List[str], **kwargs) -> torch.Tensor:
        """Process a shard of data using the vLLM model."""
        # TODO: Implement vLLM processing
        pass


class RuleBase:
    pass


def create_model(
    model_type: str,
    model_path: str,
    device: torch.device = None,
    max_batch_size: int = 64
) -> Union[ClassificationModel, APIModel, VLLMModel]:
    """Factory function to create appropriate model based on type."""
    if model_type == "classification_model":
        return ClassificationModel(model_path, device, max_batch_size)
    elif model_type == "api_model":
        return APIModel(model_path)
    elif model_type == "llm_model":
        return VLLMModel(model_path, str(device))
    else:
        raise ValueError(f"Unknown model type: {model_type}")