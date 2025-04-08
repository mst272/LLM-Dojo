"""
Model implementations for different types of scoring models.
Contains classes and functions for classification models, API models, and local vLLM models.
"""
import asyncio
from typing import List, Dict, Tuple, Union
import torch
import numpy as np
from datasets import Dataset


class APIModel:
    """API-based model for scoring (e.g., GPT-3.5, GPT-4)."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

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