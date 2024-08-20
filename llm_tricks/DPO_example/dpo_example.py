import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 1、加载模型与tokenizer
model_path = r'D:\GithubProject\LLM\download_llm\qwen\Qwen1___5-0___5B'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 2、处理数据


















if __name__ == "__main__":
    pass
