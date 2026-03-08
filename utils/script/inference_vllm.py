from vllm import LLM, SamplingParams
import os
import torch
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
model_name_or_path = 'DeepSeek-R1-Distill-Qwen-32B' # model path
llm = LLM(
    model=model_name_or_path,
    max_model_len=8192,
    device='cuda',
    dtype=torch.bfloat16,
    tensor_parallel_size=8   # CUDA_VISIBLE_DEVICES 数量
)

prompt = "请帮我生成一个关于夏天的诗歌。"


TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path)
messages = [
    {"role": "user", "content": prompt}
]
text = TOKENIZER.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

prompts = [text]

sampling_params = SamplingParams(
    max_tokens=8192,
    top_p=0.9,
    top_k=1,
    temperature=0.0,
    repetition_penalty=1.0,
)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:\n{prompt}")
    print(f"Generated text:\n {generated_text}")
