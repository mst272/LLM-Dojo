from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# base模型和lora训练后保存模型的位置
base_model_path = 'download_llm/LLM-Research/Phi-3-mini-128k-instruct'
lora_path = '/LLM-out/checkpoint-616'
# 合并后整个模型的保存地址
merge_output_dir = 'merged_lora_model'

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

lora_model = PeftModel.from_pretrained(base_model, lora_path)
model = lora_model.merge_and_unload()

if merge_output_dir:
    model.save_pretrained(merge_output_dir)
    tokenizer.save_pretrained(merge_output_dir)