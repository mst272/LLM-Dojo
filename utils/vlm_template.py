from typing import Any, Dict, List
import torch
from loguru import logger
from PIL import Image


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