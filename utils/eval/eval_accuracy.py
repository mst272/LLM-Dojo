from typing import List, Dict
import evaluate
from abc import ABC, abstractmethod


# 添加新的评估方式，只需要在此文件中创建新的评估指标类
class BaseMetric(ABC):
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        pass

    @abstractmethod
    def extract_generation(self):
        pass


class CodeEvalMetric(BaseMetric):
    def __init__(self, metric_path: str = './metrics/code_eval'):
        self.metric = evaluate.load(metric_path)

    def compute(self, predictions: List[List[str]], references: List[str]) -> float:
        """
        Compute code evaluation metrics

        Args:
            predictions: List of list of predicted code strings
            references: List of reference code strings

        Returns:
            float pass@1
        """
        pass_at_k, results = self.metric.compute(
            predictions=predictions,
            references=references,
            k=[1]
        )
        return float(pass_at_k["pass@1"])


class EmMetric(BaseMetric):
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        pass

    def extract_generation(self):
        pass
