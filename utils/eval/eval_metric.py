import os
import re
from typing import List, Dict, Union, Type, Optional

import psutil

import evaluate
from abc import ABC, abstractmethod
import gc
from datasets import Dataset

from utils.eval.configs import EvaluationConfig
from contextlib import contextmanager


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
    def __init__(self, metric_path: str = './utils/eval/metrics/code_eval'):
        # self.metric = evaluate.load(metric_path)
        self.metric_path = metric_path
        self.memory_threshold = 0.8  # 80% memory usage threshold

    def _check_memory_usage(self) -> Optional[float]:
        """Monitor memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f}MB ({memory_percent:.1f}%)")

        return memory_percent

    @contextmanager
    def load_metric(self):
        """Context manager for loading and cleaning up metric with memory monitoring"""
        try:
            memory_percent = self._check_memory_usage()
            if memory_percent > self.memory_threshold*100:
                print(f"Warning: High memory usage detected: {memory_percent:.1f}%")
                gc.collect()

            metric = evaluate.load(self.metric_path)
            yield metric
        finally:
            del metric
            gc.collect()
            self._check_memory_usage()

    def compute(self, predictions: List[List[str]], references: List[str]) -> float:
        with self.load_metric() as metric:
            try:
                pass_at_k, results = metric.compute(
                    predictions=predictions,
                    references=references,
                    k=[1]
                )
                return float(pass_at_k["pass@1"])
            except Exception as e:
                print(f"Error during computation: {str(e)}")
                raise e


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
            prompts = [test_dataset['prompt'] for test_dataset in test_datasets]
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            res = [tokenizer.apply_chat_template(message, tokenize=False,add_generation_prompt=True) for message in messages]
            return res

        else:
            prompts = [test_dataset['prompt'] for test_dataset in test_datasets]
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
