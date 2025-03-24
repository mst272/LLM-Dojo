from dataclasses import dataclass, asdict
import os
import json
from typing import Dict, List

from datasets import load_dataset
from vllm import LLM, SamplingParams


@dataclass
class Args:
    dataset_name: str = './test.jsonl'
    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    save_filename: str = "completions.jsonl"
    skill: str = "chat"
    mode: str = "generation"  # Can be "generation" or "judgment"


@dataclass
class GenerationArgs:
    num_completions: int = 3
    temperature: float = 0.8
    response_length: int = 4096
    top_p: float = 0.9
    tensor_parallel_size: int = 1


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


def generate_with_vllm(model_name_or_path: str, dtype: str, prompt_token_ids: List[int], gen_args: GenerationArgs):
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=gen_args.tensor_parallel_size,
        max_model_len=gen_args.response_length,
        dytype=dtype
    )

    # filter out prompts which are beyond the model's max token length
    max_model_len = llm.llm_engine.scheduler_config.max_model_len
    prompt_token_ids_len = len(prompt_token_ids)
    prompt_token_ids = [item for item in prompt_token_ids if len(item) < max_model_len]
    if len(prompt_token_ids) != prompt_token_ids_len:
        print(f"Filtered out {prompt_token_ids_len - len(prompt_token_ids)} prompts which exceeds max token length")

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            n=gen_args.num_completions,
            temperature=gen_args.temperature,
            top_p=1.0,
            max_tokens=gen_args.response_length,
            include_stop_str_in_output=True,
        ),
    )

    return [
        {
            "outputs": [asdict(out) for out in output.outputs],
            "prompt": output.prompt,
            "prompt_logprobs": output.prompt_logprobs,
            "metrics": output.metrics,
        }
        for output in outputs
    ]


def main(args: Args, gen_args: GenerationArgs):
    dataset = load_dataset(data_files=args.dataset_name, path='json')  # 适配jsonl格式




















