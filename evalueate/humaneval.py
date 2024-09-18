import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import os
from utils import language_settings, extract_generation_code
from evaluation import evaluate_functional_correctness
from args import EvaluateArgs


def build_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())


def generate_one(example, lang, tokenizer, model, args):
    prompt = build_instruction(language_settings[lang]['full_name'], example['prompt'])
    # 适配模型Chat Template
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

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


def generate_main(args):
    model_name_or_path = args.model_name_or_path
    saved_path = args.output_path

    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    torch_dtype = torch.bfloat16 if args.torch_dtype == 'bf16' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    examples = [json.loads(x) for x in open(args.data_file) if x.strip()]
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex, args.language, tokenizer, model, args)
        generated_examples.append(gen_example)

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))

    result = evaluate_functional_correctness(
        input_file=saved_path,
        n_workers=8,
        timeout=3.0,
        k=1
    )
    print(result, model_name_or_path)


def evaluate_only(args):
    pass


if __name__ == '__main__':
    parser = HfArgumentParser((EvaluateArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)