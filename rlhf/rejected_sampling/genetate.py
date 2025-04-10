from collections import defaultdict
from dataclasses import dataclass, asdict
import os
import json
from typing import Dict, List

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rlhf.utils.util import ArgumentParserPlus


@dataclass
class Args:
    dataset_name: str = './test.jsonl'  # 数据集
    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    save_filename: str = "completions.jsonl"
    auto_adapt: bool = True  # if apply chat template
    system: str = ''  # chat template default


@dataclass
class GenerationArgs:
    num_completions: int = 3
    temperature: float = 0.8
    response_length: int = 4096
    top_p: float = 0.9
    tensor_parallel_size: int = 1
    dtype: torch.dtype = torch.bfloat16


def load_datasets(data_files: str, shuffle: bool):
    """
    读取数据集，单jsonl文件或者目录
    """
    if os.path.isfile(data_files):
        # 如果是单个文件，直接读取
        if not data_files.endswith('.jsonl'):
            raise ValueError(f"文件 '{data_files}' 不是JSONL文件")
        datasets = load_dataset("json", data_files=data_files)
    else:
        # 如果是目录，读取所有JSONL文件
        datasets = []
        jsonl_files = [f for f in os.listdir(data_files) if f.endswith('.jsonl')]
        for file_name in jsonl_files:
            dataset = load_dataset("json", data_files=file_name)
            datasets.append(dataset)
        datasets = concatenate_datasets(datasets)
    if shuffle:
        datasets = datasets.shuffle(seed=42)
    return datasets['train']


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


def save_jsonl_in_chunks_to_files(base_filename: str, table: Dict[str, List], chunksize: int):
    """
    将字典数据按指定的 chunksize 分块保存为多个 JSONL 文件。

    Args:
        base_filename: 保存的文件名的基本名称（不包含 chunk 编号）。
        table: 包含数据的字典，其中 values 是等长的列表。
        chunksize: 每个 chunk 文件保存的行数。
    """
    first_key = list(table.keys())[0]
    num_rows = len(table[first_key])
    os.makedirs(os.path.dirname(base_filename), exist_ok=True)
    chunk_number = 0
    for i in range(0, num_rows, chunksize):
        chunk_number += 1
        save_filename = f"{base_filename}_chunk_{chunk_number}.jsonl"
        with open(save_filename, "w") as outfile:
            for j in range(i, min(i + chunksize, num_rows)):
                json.dump({key: table[key][j] for key in table}, outfile)
                outfile.write("\n")


def generate_with_vllm(model_name_or_path: str, prompt_token_ids: List[int], gen_args: GenerationArgs):
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=gen_args.tensor_parallel_size,
        max_model_len=gen_args.response_length,
        dytype=gen_args.dtype
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


def tokenize(dataset: Dataset, auto_adapt: bool, system: str, tokenizer):
    def tokenize_fn(row):
        answer = row['answer'] if 'answer' in row else row['messages'][1]['content']
        prompt = row['prompt'] if 'prompt' in row else row['messages'][0]['content']
        messages = [
            {"role": "user", "content": prompt}
        ]
        if system is not None:
            messages.append({"role": "system", "content": system})
        outputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        return {"input_ids": outputs, "prompt": prompt, "answer": answer}

    def tokenize_fn_origin(row):
        prompt = row['prompt'] if 'prompt' in row else row['messages'][0]['content']
        answer = row['answer'] if 'answer' in row else row['messages'][1]['content']
        outputs = tokenizer.encode(prompt)
        return {"input_ids": outputs, "prompt": prompt, "answer": answer}

    return dataset.map(
        tokenize_fn if auto_adapt else tokenize_fn_origin,
        desc="Tokenizing and reformatting rejected sampling data",
    )


def main(args: Args, gen_args: GenerationArgs):
    dataset = load_datasets(data_files=args.dataset_name, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = tokenize(dataset=dataset, auto_adapt=args.auto_adapt, system=args.system, tokenizer=tokenizer)
    prompt_token_ids = dataset['input_ids']
    outputs = generate_with_vllm(args.model_name_or_path, prompt_token_ids, gen_args)

    # Assuming we generate n=3 completions per prompt; the outputs will look like:
    # prompt | completions
    # -------|------------
    # q1     | a1
    # q1     | a2
    # q1     | a3
    # q2     | a1
    # ...
    table = defaultdict(list)
    num_prompt_with_identical_completions = 0
    for output, answer, prompt in zip(outputs, dataset["answer"], dataset['prompt']):
        # if the model completions are exactly the same across all completions per prompt, we can skip this
        if len(set(tuple(item["text"]) for item in output["outputs"])) == 1:
            num_prompt_with_identical_completions += 1
            continue

        for item in output["outputs"]:
            new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": item["text"]}]
            table["messages"].append(new_messages)
            table["model_completion"].append(item["text"])
            table["reference_completion"].append(answer)

    print(f"Number prompts with identical completions: {num_prompt_with_identical_completions}")
    save_jsonl(args.save_filename, table)


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, GenerationArgs))
    main(*parser.parse())
