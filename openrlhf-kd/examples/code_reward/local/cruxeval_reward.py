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


def extract_answer_block(text: str) -> str:
    """
    Extract content inside `[ANSWER]...[/ANSWER]` block from model output.
    Returns 'raise' when no block found to trigger execution error.
    """
    matches = re.findall(r'\[ANSWER\](.*?)\[/ANSWER\]', text, re.DOTALL)
    return matches[-1].strip() if matches else 'raise'


def reward_func(queries, prompts, labels, **kwargs):
    """
    Compute reward based on whether extracted `assert` test code executes successfully.

    Args:
        queries: Model-generated responses (List[str]).
        labels: Code to be tested (List[str]).
    """
    metric = _get_metric(kwargs.get("code_eval_metric", None))
    rewards = []

    for generated_text, original_code in zip(queries, labels):
        try:
            test_block = extract_answer_block(generated_text)
            
            # CruxEval requires `assert` statement in generated test code
            if 'assert' in test_block:
                out_pass, _ = metric.compute(predictions=[[original_code]], references=[test_block])
                is_correct = out_pass.get("pass@1", 0.0) >= 1.0
                rewards.append(float(is_correct))
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"Code evaluation error: {e}")
            rewards.append(0.0)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {"dummy_scores": rewards_tensor}
    }
