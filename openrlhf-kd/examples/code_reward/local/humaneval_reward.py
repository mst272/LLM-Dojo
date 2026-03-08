import os
import re
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRIC_PATH = os.path.join(_SCRIPT_DIR, "..", "code_eval")
_METRIC = None


def _get_metric(code_eval_metric=None):
    """
    Get or load the code_eval metric.
    Prefer externally injected metric (for mixed training), otherwise load local evaluator.
    """
    global _METRIC
    if code_eval_metric is not None:
        return code_eval_metric

    if _METRIC is None:
        # Configure Hugging Face environment to allow code evaluation
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        hf_home = os.environ.setdefault("HF_HOME", "/tmp/hf")
        os.environ.setdefault("HF_MODULES_CACHE", os.path.join(hf_home, "modules"))

        import evaluate
        _METRIC = evaluate.load(METRIC_PATH)
        
    return _METRIC


def extract_code_block(code):
    """
    Extract the last code block from markdown-style code.

    Returns:
        str: Content of last ```python ...``` block if found, otherwise 'raise'.
    """
    code_pattern = r'```(?:python|go|ts|php|csharp|bash|javascript|cpp|cs|java|typescript|js|sh)(.*?)```'
    code_match = re.findall(code_pattern, code, re.DOTALL)

    if code_match:
        return code_match[-1].strip()
    else:
        return 'raise'


def reward_func(queries, prompts, labels, **kwargs):
    """
    Compute reward based on whether extracted code passes all test cases.

    Args:
        queries: Model-generated responses (List[str]).
        labels: Test cases for evaluation (List[str]).
    """
    metric = _get_metric(kwargs.get("code_eval_metric", None))

    rewards = []

    for generated_text, test_cases in zip(queries, labels):
        try:
            code = extract_code_block(generated_text)
            
            # evaluate.compute returns: ({'pass@1': 1.0}, {details...})
            out_pass, _ = metric.compute(predictions=[[code]], references=[test_cases])
            
            # pass@1 >= 1.0 means all test cases passed
            is_correct = out_pass.get("pass@1", 0.0) >= 1.0
            rewards.append(float(is_correct))
            
        except Exception as e:
            # Catch syntax errors, timeouts, or other exceptions
            print(f"Code evaluation error: {e}")
            rewards.append(0.0)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {"dummy_scores": rewards_tensor}
    }
