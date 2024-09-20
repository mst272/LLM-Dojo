import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation import evaluate_functional_correctness


def generate_one(example, tokenizer, model, args, task):
    prompt = task.build_instruction(example)
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

    output = tokenizer.decode(outputs[0][:], skip_special_tokens=True)
    example['output'] = output

    return task.generation_code_process(example)


def generate_main(args, task):
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
        gen_example = generate_one(ex, tokenizer, model, args, task)
        generated_examples.append(gen_example)

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))

    result = task.evaluate_function(saved_path,args)
    save_metrics(args, result)
    print(result, model_name_or_path)


def evaluation_only(args, task):
    result = task.evaluate_function(args.evaluate_data_path, args)
    save_metrics(args, result)
    print(result, args.model_name_or_path)


def save_metrics(args, result):
    args_dict = args.__dict__
    with open(args.save_metrics_path, 'w', encoding='utf-8') as fw:
        fw.write(json.dumps(result) + '\n')
        fw.write(json.dumps(args_dict) + '\n')

