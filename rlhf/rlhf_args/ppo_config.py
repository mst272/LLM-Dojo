from dataclasses import dataclass, field
from typing import Optional
from trl.trainer.ppov2_trainer import PPOv2Config


@dataclass
class PPOConfig(PPOv2Config):

    # TrainingArguments的相关参数
    train_data_path: Optional[str] = field(default='./', metadata={"help": "训练集路径"})

