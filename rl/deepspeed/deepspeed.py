
from abc import ABC

class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """
    def setup_distributed():
        pass