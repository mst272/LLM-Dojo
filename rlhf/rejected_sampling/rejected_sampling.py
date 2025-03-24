from dataclasses import dataclass, field
from typing import List
import os


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

    # upload config
    hf_repo_id: str = os.path.basename(__file__)[: -len(".py")]
    hf_repo_id_scores: str = os.path.basename(__file__)[: -len(".py")] + "_scores"
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    add_timestamp: bool = True
