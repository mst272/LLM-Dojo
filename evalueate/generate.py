import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import os
from utils import language_settings, extract_generation_code
from evaluation import evaluate_functional_correctness
from args import EvaluateArgs


def build_instruction(prompt: str):
    """
    根据模型构建合适的指令
    """
    return prompt


def generate_one(example, lang, tokenizer, model, args):
    # prompt = build_instruction(language_settings[lang]['full_name'], example['prompt'])
    # 适配模型Chat Template
    # inputs = tokenizer.apply_chat_template(
    #     [{'role': 'user', 'content': prompt}],
    #     return_tensors="pt",
    #     add_generation_prompt=True
    # ).to(model.device)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    stop_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(
        "<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"

    outputs = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
        pad_token_id=stop_id,
        eos_token_id=stop_id
    )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    example['output'] = output

    return extract_generation_code(example, lang_code=lang)
