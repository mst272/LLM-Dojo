from typing import Any, Dict, List
import torch
from loguru import logger
from PIL import Image


class SftDataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出最大长度
        length = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 每个batch中的最大长度
        max_batch_length = min(self.max_length, max(length))

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []

        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None,and now continue')
                continue
            padding_len = max_batch_length - len(input_ids)
            # 开始padding
            input_ids += [self.pad_token_id] * padding_len
            attention_mask += [0] * padding_len
            target_mask += [0] * padding_len
            # 开始截断
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            target_mask = target_mask[:self.max_length]
            # 将本批次全部加入列表
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        # 计算损失时忽略
        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs


# todo: 针对不同VLM进行输入message处理
def llava_template_process(questions: List[str], answers: List[str]) -> List[Dict]:
    converted_data = []
    for question, answer in zip(questions, answers):
        user_content = [{'index': None, 'text': question, 'type': 'text'}]
        assistant_content = [{'index': None, 'text': answer, 'type': 'text'}]

        converted_data.append({'content': user_content, 'role': 'user'})
        converted_data.append({'content': assistant_content, 'role': 'assistant'})
    image_dict = {'index': 0, 'text': None, 'type': 'image'}
    converted_data[0]['content'].append(image_dict)
    return converted_data


def qwen_template_process(questions: List[str], answers: List[str]):
    pass


class VlmQaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            standard_example = llava_template_process(example[0], example[1])
            text = self.processor.apply_chat_template(
                standard_example, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            raw_image = Image.open(example[2])
            images.append(raw_image)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # 这里并没有mask question, 后续可能的扩充是设置mask question的模式。
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch['labels'] = labels

        return batch
