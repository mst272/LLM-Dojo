from dataclasses import dataclass, field
from typing import Optional, List
from trl import ModelConfig


@dataclass
class OurModelConfig(ModelConfig):
    """
       Arguments which define the model and tokenizer to load.
    """

    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "模型加载路径"})
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code when loading a model."})

    # 下面参数是关于lora的配置， use_peft=True 表示启用lora
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT or not for training."})
    torch_dtype: Optional[str] = field(default='bfloat16', metadata={"help": "4位精度计算的数据类型", "choices": ["auto", "bfloat16", "float16", "float32"]})
    lora_r: Optional[int] = field(default=16, metadata={"help": "LoRA R value."})
    lora_alpha: Optional[int] = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_task_type: str = field(default="SEQ_CLS", metadata={"help": "The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"})

    # 下列参数是关于qlora的配置， load_in_4bit=True 则表示使用qlora。若为 False 则后面参数不需要设置。
    load_in_4bit: bool = field(default=False, metadata={"help": "是否使用qlora"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "4位精度量化的类型。这里设置为nf4 表示使用nf4量化类型。"})
    use_bnb_nested_quant: bool = field(default=True, metadata={"help": "是否使用双精度量化。如果设置为True，则使用双精度量化"})
