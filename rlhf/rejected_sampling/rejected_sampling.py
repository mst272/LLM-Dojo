import asyncio
import copy
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    PreTrainedTokenizer, HfArgumentParser
from models import create_model


@dataclass
class Args:
    use_model_type: str = 'llm_model'  # ['api_model','classification_model', 'llm_model]
    model_names_or_paths: List[str] = field(default_factory=lambda: ["gpt-4"])
    input_filename: str = "completions.jsonl"
    save_filename: str = "rejected_sampling_completions.jsonl"
    save_filename_scores: str = "completion_scores.jsonl"
    num_completions: int = 1
    max_forward_batch_size: int = 64
    num_gpus: int = 1  # New argument for specifying the number of GPUs
    mode: str = "judgement"
    skill: str = "chat"  # [chat, code, code-chat]
    include_reference_completion_for_rejection_sampling: bool = True


@dataclass
class GenerationArgs:
    num_completions: int = 1
    temperature: float = 0.8
    response_length: int = 2048
    top_p: float = 0.9
    tensor_parallel_size: int = 1
    dtype: torch.dtype = torch.bfloat16


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


def load_completions(filename: str) -> List[dict]:
    """Load completions from JSONL file."""
    with open(filename) as f:
        return [json.loads(line) for line in f]


def get_model_scores(
        model_path: str,
        args: Args,
        shards: List[List[str]]
) -> torch.Tensor:
    """Get scores based on model type."""
    results = []
    # API 模型和 vLLM 模型直接处理
    model = create_model(args.use_model_type, model_path)
    for shard in shards:
        results.append(model.get_score(shard))

    return torch.cat(results)


def main():
    # Parse arguments
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]

    # Load and preprocess data
    completions = load_completions(args.input_filename)

    # process: include the reference completion in the completions for efficient rejection sampling
    if args.include_reference_completion_for_rejection_sampling:
        new_completions = []
        for i in range(len(completions)):
            if i % args.num_completions == 0:
                reference_completion = copy.deepcopy(completions[i])
                reference_completion["messages"][-1]["content"] = reference_completion["reference_completion"]
                reference_completion["model_completion"] = reference_completion["reference_completion"]
                new_completions.append(reference_completion)
            new_completions.append(completions[i])
        completions = new_completions
        actual_num_completions = args.num_completions + 1  # we have added the reference completion

    # rejected sampling by api models

    # Save results

if __name__ == "__main__":
    main()
