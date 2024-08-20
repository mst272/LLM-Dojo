import torch
from torch.utils.data import Dataset
import json


class RlhfDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as file:
            data_list = file.readlines()
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']

        chosen_full_text = f"{prompt}\n\n### Response:\n{chosen}"
        rejected_full_text = f"{prompt}\n\n### Response:\n{rejected}"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_full_tokens = self.tokenizer.encode(chosen_full_text, add_special_tokens=False)
        rejected_full_tokens = self.tokenizer.encode(rejected_full_text, add_special_tokens=False)

        input = {
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            }
        return input

    def __len__(self):
        return len(self.data_list)


def data_collate(
        batch,
        pad_token_id=50256,
        allowed_max_length=None,
        mask_prompt_tokens=True,
        device="cpu"
):
    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []

    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # Set mask for all padding tokens to False
            mask[len(sequence):] = False

            # Set mask for all input tokens to False
            # +2 sets the 2 newline ("\n") tokens before "### Response" to False
            if mask_prompt_tokens:
                mask[:prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    return batch_data
