from typing import List, Dict, Any


# 基础模板处理类
class TemplateProcessor:
    def process(self, questions: List[str], answers: List[str]) -> Any:
        raise NotImplementedError("Subclasses must implement `process` method.")


# LLaVA 模板处理类
class LlavaTemplateProcessor(TemplateProcessor):
    NAME = 'LlavaProcessor'

    def process(self, questions: List[str], answers: List[str]) -> List[Dict]:
        converted_data = []
        for question, answer in zip(questions, answers):
            user_content = [{'index': None, 'text': question, 'type': 'text'}]
            assistant_content = [{'index': None, 'text': answer, 'type': 'text'}]

            converted_data.append({'content': user_content, 'role': 'user'})
            converted_data.append({'content': assistant_content, 'role': 'assistant'})
        image_dict = {'index': 0, 'text': None, 'type': 'image'}
        converted_data[0]['content'].append(image_dict)
        return converted_data


# Qwen2VL 模板处理类可与llava相同
class Qwen2VLTemplateProcessor(LlavaTemplateProcessor):
    NAME = 'Qwen2VLProcessor'
