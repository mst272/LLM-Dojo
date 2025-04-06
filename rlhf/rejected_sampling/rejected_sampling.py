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
    skill: str = "chat"
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


def majority_vote(model_offsets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Get majority vote across models.

    model_offsets: offsets returned by each model. each tensor is of shape (n_prompts,) indicating best/worst completion offset per prompt
    """
    # Determine the number of samples
    num_samples = next(iter(model_offsets.values())).size(0)
    # Initialize tensor to store the majority votes
    votes = torch.zeros(num_samples, dtype=torch.long)

    # Tally the votes and determine the majority vote for each sample
    for i in range(num_samples):
        # Collect votes from all models for the current sample
        sample_votes = [offsets[i].item() for offsets in model_offsets.values()]
        # Determine the most common vote
        counter = Counter(sample_votes)
        # Try to get the majority vote, but if all models disagree, we randomly choose one
        votes[i] = counter.most_common(1)[0][0] if len(model_offsets) != len(counter) else \
            sample_votes[np.random.randint(len(sample_votes))]
    return votes


def get_model_scores(
        model_path: str,
        args: Args,
        shards: List[List[str]]
) -> torch.Tensor:
    """Get scores based on model type."""
    results = []

    if args.use_model_type == 'classification_model':
        # 使用多进程处理 classification model
        with mp.Pool(args.num_gpus) as pool:
            for i, shard in enumerate(shards):
                model = create_model(
                    args.use_model_type,
                    model_path,
                    torch.device(f"cuda:{i}"),
                    args.max_forward_batch_size
                )
                results.append(
                    pool.apply_async(model.get_score, (shard,))
                )
            results = [r.get() for r in results]
    else:
        # API 模型和 vLLM 模型直接处理
        model = create_model(args.use_model_type, model_path)
        for shard in shards:
            results.append(model.get_score(shard))

    return torch.cat(results)


def main():
    # Parse arguments
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]

    # Setup multiprocessing
    mp.set_start_method("spawn", force=True)

    # Load and preprocess data
    completions = load_completions(args.input_filename)

    # process: include the reference completion in the completions for efficient rejection sampling
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

    # Split the data into shards
    shard_size = len(completions) // args.num_gpus
    shards = [completions[i: i + shard_size] for i in range(0, len(completions), shard_size)]

    # Process shards in parallel
    best_offsets_per_model = {}
    worst_offsets_per_model = {}
    reference_completion_scores_per_model = {}
    for model_name_or_path in args.model_names_or_paths:
        results = []
        # if use openai
        # if use_api_model:  #todo: model_type
        if "gpt-3.5" in model_name_or_path or "gpt-4" in model_name_or_path:
            # when using LLM as a judge, num_gpus here refers to the number of shards as we query an API and we don't
            # use GPUs
            for i in range(args.num_gpus):
                results.append(process_shard_api(model_name_or_path, args, shards[i]))
            scores = []
            for result in results:
                scores.append(result)
        elif args.use_model_type == 'llm_model':
            pass
        else:  # classification model
            with mp.Pool(args.num_gpus) as pool:  # NOTE: the `result.get()` need to live in this `mp.Pool` context
                for i in range(args.num_gpus):
                    results.append(
                        pool.apply_async(process_shard_classification_model, (i, model_name_or_path, args, shards[i])))
                # Collect results
                scores = []
                for result in results:
                    scores.append(result.get())

        # Combine scores from all GPUs
        scores = torch.cat(scores)
        scores_per_prompt = scores.reshape(-1, actual_num_completions)  # (n_prompts, n_completions)
        reference_completion_scores = scores_per_prompt[:, 0]
        reference_completion_scores_per_model[model_name_or_path] = reference_completion_scores.tolist()

        if not args.include_reference_completion_for_rejection_sampling:
            scores_per_prompt = scores_per_prompt[:, 1:]
            scores = scores_per_prompt.flatten()
            completions = [completions[i] for i in range(len(completions)) if i % actual_num_completions != 0]
            actual_num_completions -= 1

        assert len(completions) == len(scores)
        # Rejection sampling
        for i in range(len(scores)):
            if "score" not in completions[i]:
                completions[i]["score"] = {}
            completions[i]["score"][model_name_or_path] = scores[i].item()
            if "reference_completion_score" not in completions[i]:
                completions[i]["reference_completion_score"] = {}
            completions[i]["reference_completion_score"][model_name_or_path] = reference_completion_scores[
                i // actual_num_completions
                ].item()

        best_indices = torch.argmax(scores_per_prompt, dim=1)  # (n_prompts, 1) --> (n_prompts, )
        worst_indices = torch.argmin(scores_per_prompt, dim=1)  # (n_prompts, 1) --> (n_prompts, )
        best_indices_offset = (
                torch.arange(0, len(best_indices) * actual_num_completions, actual_num_completions) + best_indices
        )
        best_offsets_per_model[model_name_or_path] = best_indices_offset

        worst_indices_offset = (
                torch.arange(0, len(worst_indices) * actual_num_completions, actual_num_completions) + worst_indices
        )
        worst_offsets_per_model[model_name_or_path] = worst_indices_offset

    # Majority vote
    best_indices_offset = majority_vote(best_offsets_per_model)
    worst_indices_offset = majority_vote(worst_offsets_per_model)

    best_completions = [completions[i] for i in best_indices_offset]
    worst_completions = [completions[i] for i in worst_indices_offset]

    # Save results
    table = defaultdict(list)
    for i in range(len(best_completions)):
        table["chosen"].append(best_completions[i]["messages"])
        table["rejected"].append(worst_completions[i]["messages"])
        table["reference_completion"].append(worst_completions[i]["reference_completion"])
        table["reference_completion_score"].append(
            {key: reference_completion_scores_per_model[key][i] for key in reference_completion_scores_per_model}
        )
        assert worst_completions[i]["messages"][:-1] == best_completions[i]["messages"][:-1]
        table["chosen_score"].append(best_completions[i]["score"])
        table["rejected_score"].append(worst_completions[i]["score"])
    save_jsonl(args.save_filename, table)

    table_scores = defaultdict(list)
    keys = list(completions[0].keys())
    for i in range(len(completions)):
        for key in keys:
            table_scores[key].append(completions[i][key])
    save_jsonl(args.save_filename_scores, table_scores)


if __name__ == "__main__":
    main()
