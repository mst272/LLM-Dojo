from typing import List, Dict, Any


def get_key(model):
    """获取模型的key"""
    key_set = set()
    for name, param in model.named_parameters():
        key_set.add(name.split('.')[0])
    return key_set


# 基础模板处理类
class TemplateProcessor:
    def __init__(self):
        # model 参数名称
        self.model_key = {
            "visual": 'visual',
            "llm": 'llm',
            "projector": 'projector'
        }

    def process(self, questions: List[str], answers: List[str]) -> Any:
        raise NotImplementedError("Subclasses must implement `process` method.")


# LLaVA 模板处理类
class LlavaTemplateProcessor(TemplateProcessor):
    NAME = 'LlavaProcessor'

    def __init__(self):
        super().__init__()
        self.model_key['visual'] = 'vision_tower'
        self.model_key['llm'] = 'language_model'
        self.model_key['projector'] = 'multi_modal_projector'

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

    def __init__(self):
        super().__init__()
        # model 参数名称
        self.model_key['visual'] = 'visual'
        self.model_key['llm'] = 'model'
        self.model_key['projector'] = 'lm_head'


if __name__ == '__main__':
    qwen = Qwen2VLTemplateProcessor()
    print(qwen.model_key)
