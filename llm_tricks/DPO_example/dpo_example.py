import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
from dataset import RlhfDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from loss import DPOLoss

# 1、加载模型与tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = r'D:\GithubProject\LLM\download_llm\qwen\Qwen1___5-0___5B'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
ref_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
model.to(device)
ref_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

# 2、处理数据
# 加载数据
data_file = './unsloth_dpo.jsonl'
# Dataset详细逻辑可看进入RlhfDataset实现
dataset = RlhfDataset(data_file, tokenizer)
# 划分训练集验证集
train_size = int(len(dataset) * 0.85)  # 85% for training
val_size = len(dataset) - train_size  # Remaining for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 编写batch批次的padding及mask处理函数
IGNORE_INDEX = False


def data_collector(batch, pad_token, device, max_length=None, if_mask_prompt=True):
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []
    }

    # 判断长度及padding
    max_length_common = 0
    for key in ["chosen", "rejected"]:
        current_max = max(len(item[key]) for item in batch)
        max_length_common = max(max_length_common, current_max)

    # 转为torch tensor并padding,决定是否对prompt进行mask
    for item in batch:
        prompt = torch.tensor(item['prompt'])
        batch_data['prompt'].append(prompt)

        for key in ["chosen", "rejected"]:
            out = item[key]
            out_padding = out + [pad_token] * (max_length_common - len(out))
            mask = torch.ones(len(out_padding)).bool()

            # padding部分的mask设置为 IGNORE_INDEX
            mask[len(out):] = IGNORE_INDEX

            if if_mask_prompt:
                mask[:prompt.shape[0] + 2] = IGNORE_INDEX
            batch_data[key].append(torch.tensor(out_padding))
            batch_data[f"{key}_mask"].append(mask)

    # 进行最大长度截断
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])
        if max_length is not None:
            tensor_stack = tensor_stack[:, :max_length]
        # 将tensor移到对应的device
        batch_data[key] = tensor_stack.to(device)
    return batch_data


# 3、开始计算DPO(或其他)的损失函数
loss_fn = DPOLoss()


if __name__ == "__main__":
    pass
