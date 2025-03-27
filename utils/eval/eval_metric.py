import re
from typing import List, Dict, Union, Type
import evaluate
from abc import ABC, abstractmethod

from datasets import Dataset

from utils.eval.configs import EvaluationConfig


# 添加新的评估方式，只需要在此文件中创建新的评估指标类

class BaseMetric(ABC):
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        pass

    @abstractmethod
    def extract_generation(self, completions: List[str]):
        pass

    def get_prompts(self, test_datasets: Dataset, tokenizer, prompts_apply_chat) -> List[str]:
        """
        The default get prompts(questions) from the test data.

        Args:
            test_datasets: dataset must include a `"prompt"` column containing the prompts for generating completions.
            tokenizer:
            prompts_apply_chat: :
        Returns:
            prompts, ['hellow','why']
        """
        if prompts_apply_chat:
            pass
        else:
            prompts = [test_dataset['prompt'] for test_dataset in test_datasets]
        return prompts

    @staticmethod
    def get_labels(test_datasets: Dataset):
        labels = [test_dataset['label'] for test_dataset in test_datasets]
        return labels


class CodeEvalMetric(BaseMetric):
    def __init__(self, metric_path: str = './metrics/code_eval'):
        self.metric = evaluate.load(metric_path)

    def compute(self, predictions: List[List[str]], references: List[str]) -> float:
        """
        Compute python code evaluation metrics

        Args:
            predictions: A list of lists, where each sublist contains predicted code strings.
            references: List of reference code strings.

        Returns:
            float pass@1
        """
        pass_at_k, results = self.metric.compute(
            predictions=predictions,
            references=references,
            k=[1]
        )
        return float(pass_at_k["pass@1"])

    def extract_generation(self, completions: List[str]):
        """
        extract generation and process data to compute[predictions] format

        Args:
            completions List[str]: 输入字符串。

        Returns:
            self.compute[predictions] format. For example, there is List[List[str]]
        """
        def extract(code: str, index: int):
            # Look for code blocks
            code_pattern = r'```(?:python|go|javascript|java|bash|js|cpp|cs|php)(.*?)```'
            code_match = re.findall(code_pattern, code, re.DOTALL)
            if code_match:
                # If code block exists, return its content (excluding the ``` markers)
                return code_match[-1].strip()
            else:
                # If no code block, return the solution content directly
                return str(index)

        generations = [[extract(c, i)] for i, c in enumerate(completions)]
        return generations

    def get_prompts(self, test_datasets: Dataset, tokenizer, prompts_apply_chat) -> List[str]:
        if prompts_apply_chat:
            pass
        else:
            pre_instruct = '补全下面代码，将最终题目和答案返回在代码框中\n'
            prompts = [pre_instruct + test_dataset['prompt'] for test_dataset in test_datasets]
        return prompts


class ExactMatchMetric(BaseMetric):
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        pass

    def extract_generation(self, completions: List[str]):
        return completions


# todo: 未完成
class CompositeMetric(BaseMetric):
    """简化的混合指标类"""

    def __init__(self, metric_configs: List[Dict[str, Union[str, float]]]):
        self.metrics = []
        self.weights = []

        # 创建指标实例
        for config in metric_configs:
            metric_name = config['name']
            weight = config.get('weight', 1.0)

            if metric_name == 'code':
                metric = CodeEvalMetric()
            elif metric_name == 'em':
                metric = ExactMatchMetric()
            else:
                raise ValueError(f"不支持的指标类型: {metric_name}")

            self.metrics.append(metric)
            self.weights.append(weight)

    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        results = {}
        total_score = 0.0

        for metric, weight in zip(self.metrics, self.weights):
            metric_results = metric.compute(predictions, references)
            for key, value in metric_results.items():
                weighted_value = value * weight
                results[f"{metric.__class__.__name__}_{key}"] = weighted_value
                total_score += weighted_value

        results['composite_score'] = total_score
        return results

    def extract_generation(self, generate, index):
        # 使用第一个metric的提取方法
        return self.metrics[0].extract_generation(generate, index)

    def get_prompts(self, test_datasets: Dataset, tokenizer, args) -> List[str]:
        # 使用第一个metric的prompt获取方法
        return self.metrics[0].get_prompts(test_datasets, tokenizer, args)


# 定义指标类型映射字典
METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    'code': CodeEvalMetric,
    'em': ExactMatchMetric
}


def create_metric(config: EvaluationConfig) -> BaseMetric:
    """根据配置创建指标

    Args:
        config: 评估配置对象

    Returns:
        BaseMetric: 创建的指标对象

    Raises:
        ValueError: 当指定了不支持的指标类型时
    """
    # 单个指标的情况
    if isinstance(config.metrics, str):
        metric_class = METRIC_REGISTRY.get(config.metrics)
        if not metric_class:
            raise ValueError(f"不支持的指标类型: {config.metrics}")
        return metric_class()

    # 混合指标的情况
    return CompositeMetric(config.metrics)
