import json
from dataclasses import dataclass, field
from typing import List, Dict
import os
import torch.multiprocessing as mp


@dataclass
class Args:
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


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")









def main(args:Args):
    mp.set_start_method("spawn", force=True)