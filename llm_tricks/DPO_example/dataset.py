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