from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Template:
    template_name: str = field(metadata={"help": ""})
    system_format: str = field(metadata={"help": ""})
    user_format: str
    assistant_format: str
    system: str
    stop_word: str


template_dict: Dict[str, Template] = {}


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word
    )


# 开始注册
register_template(
    template_name='default',
    system_format='System: {content}\n',
    user_format='User: {content}\nAssistant: ',
    assistant_format='{content} {stop_token}',
    system=None,
    stop_word=None
)


register_template(
    template_name='qwen',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="You are a helpful assistant.",
    stop_word='<|im_end|>'
)


register_template(
    template_name='yi',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=None,
    stop_word='<|im_end|>'
)

register_template(
    template_name='gemma',
    system_format=None,
    user_format='<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n',
    assistant_format='{content}<end_of_turn>\n',
    system=None,
    stop_word='<end_of_turn>'
)

register_template(
    template_name='phi-3',
    system_format=None,
    user_format='<s>user\n{content}<|end|>\n<|assistant|>\n',
    assistant_format='{content}<|end|>\n',
    system=None,
    stop_word='<|end|>'
)

register_template(
    template_name='deepseek',
)


