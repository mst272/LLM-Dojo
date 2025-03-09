from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    max_length: int = 512
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
