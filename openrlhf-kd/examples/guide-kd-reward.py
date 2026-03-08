"""Unified Reward scheduler (API evaluation + KD).

Dispatches by datasource to corresponding evaluation logic, aggregates rewards/scores/extra_logs,
optionally computes teacher per-token logprobs (supports guided KD).

Supported datasources:
- cruxeval   -> local reward / API
- python     -> local HumanEval reward
- bigcodebench -> /eval/bigcodebench
- cpp/sh/ts/js/java/cs -> /eval/multiple

reward_func returns:
- rewards: Tensor[batch], 0.0/1.0
- scores: Tensor[batch]
- extra_logs: Dict[str, Tensor]
- teacher_log_probs: List[Tensor]
"""

import os
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch 

from examples.code_reward.api.api_eval import (
    reward_bigcodebench_api,
    make_multiple_reward,
)
from examples.kd.teacher_kd import TeacherKDManager


# =====================================================================
# code_eval metric (process-internal singleton)
# =====================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_EVAL_METRIC_PATH = os.path.join(_SCRIPT_DIR, "code_reward", "code_eval")
_CODE_EVAL_METRIC = None


def _get_code_eval_metric(metric_path: str = CODE_EVAL_METRIC_PATH):
    """Lazy-load code_eval metric."""
    global _CODE_EVAL_METRIC
    if _CODE_EVAL_METRIC is None:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ.setdefault("HF_HOME", "/tmp/hf")
        os.environ.setdefault("HF_MODULES_CACHE", os.path.join(os.environ["HF_HOME"], "modules"))
        import evaluate
        _CODE_EVAL_METRIC = evaluate.load(metric_path)
    return _CODE_EVAL_METRIC


# =====================================================================
# Lazily-loaded local reward modules
# =====================================================================

_REWARD_MODULES: Dict[str, Any] = {}


def _get_lazy_reward_func(name: str):
    """Lazy-load local reward function (humaneval / cruxeval)."""
    if name not in _REWARD_MODULES:
        if name == "humaneval":
            from examples.code_reward.local import humaneval_reward
            _REWARD_MODULES[name] = humaneval_reward.reward_func
        elif name == "cruxeval":
            from examples.code_reward.local import cruxeval_reward
            _REWARD_MODULES[name] = cruxeval_reward.reward_func
        else:
            raise ValueError(f"Unknown lazy reward module: {name}")
    return _REWARD_MODULES[name]


# =====================================================================
# datasource -> reward function mapping
# =====================================================================

DATASOURCE_TO_REWARD: Dict[str, Dict[str, Any]] = {
    # Lazily-loaded local reward
    "cruxeval": {
        "resolver": lambda: _get_lazy_reward_func("cruxeval"),
        "needs_code_eval": True,
    },
    "python": {
        "resolver": lambda: _get_lazy_reward_func("humaneval"),
        "needs_code_eval": True,
    },
    # API reward
    "bigcodebench": {
        "resolver": lambda: reward_bigcodebench_api,
        "needs_code_eval": False,
    },
    # Multi-language HumanEval
    **{
        lang: {
            "resolver": (lambda reward_fn=make_multiple_reward(lang): reward_fn),
            "needs_code_eval": False,
        }
        for lang in ("cpp", "sh", "java", "js", "ts", "cs")
    },
}

DEFAULT_DATASOURCE = "cruxeval"

# =====================================================================
# KD manager (centralize all KD configuration here)
#
# Training mode quick reference:
#   Pure RL (no KD)       -> kd_datasources="none"
#   Full KD               -> kd_datasources="all"
#   Partial KD            -> kd_datasources="python,cruxeval"
#   Guided KD (inject ans) -> guide_kd_datasources="all" or subset
#   Pure KD (skip reward)  -> skip_reward_datasources="python"
# =====================================================================

_kd_mgr = TeacherKDManager(
    # --- Teacher API ---
    default_url="http://10.222.17.214:8080/v1/completions",
    default_model="zhanlu",
    timeout=600,
    max_workers=1,  # teacher concurrency, avoid overload

    # --- Multi-teacher routing (route by datasource to different teachers) ---
    teacher_by_datasource={
        "bigcodebench": {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "cruxeval":     {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "sh":           {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "cs":           {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "cpp":          {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "java":         {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "agent_summary": {"url": "http://10.222.55.196:8080/v1/completions", "model": "zhanlu"},
    },

    # --- Routing control ---
    kd_datasources="all",                    # which ds compute teacher logprobs
    guide_kd_datasources="agent_summary",    # which ds enable guided KD
    skip_reward_datasources="agent_summary", # which ds skip reward (pure KD)

    # --- Guided KD content ---
    guide_prefix="\nHere is a reference solution:\n",
    guide_suffix="",
    tokenizer_path="",  # leave empty to fallback to default_model
)

_KNOWN_NON_REWARD_DATASOURCES = set(_kd_mgr.teacher_by_ds)
if _kd_mgr.kd_ds is not None:
    _KNOWN_NON_REWARD_DATASOURCES.update(_kd_mgr.kd_ds)
if _kd_mgr.guide_ds is not None:
    _KNOWN_NON_REWARD_DATASOURCES.update(_kd_mgr.guide_ds)
_KNOWN_NON_REWARD_DATASOURCES.update(_kd_mgr.skip_reward_ds)


def _normalize_datasource(ds: Optional[str]) -> str:
    ds_norm = (ds or DEFAULT_DATASOURCE).lower().strip()
    if ds_norm in DATASOURCE_TO_REWARD or ds_norm in _KNOWN_NON_REWARD_DATASOURCES:
        return ds_norm
    print(f"[Warning] Unknown datasource '{ds_norm}', fallback to '{DEFAULT_DATASOURCE}'")
    return DEFAULT_DATASOURCE


# =====================================================================
# Unified reward entry point
# =====================================================================

def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[str],
    datasources: Optional[List[str]] = None,
    query_token_ids: Optional[List[List[int]]] = None,
    prompt_token_lens: Optional[List[int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Unified reward entry point.

    Flow:
    1. Group by datasource, skip samples with skip-reward
    2. Call reward functions per group
    3. Build datasource statistics
    4. Compute teacher log-probs (KD routing)
    """
    batch_size = len(queries)
    if datasources is None:
        datasources = [DEFAULT_DATASOURCE] * batch_size
    normalized_datasources = [_normalize_datasource(ds) for ds in datasources]

    # --------- KD skip-reward mask ---------
    kd_skip_mask = [
        1.0 if _kd_mgr.is_skip_reward(ds) else 0.0
        for ds in normalized_datasources
    ]

    # --------- Step 1: Group by datasource ---------
    grouped: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"indices": [], "queries": [], "prompts": [], "labels": []}
    )
    for i, (q, p, lbl, ds_norm) in enumerate(zip(queries, prompts, labels, normalized_datasources)):
        if kd_skip_mask[i] > 0.5:
            continue
        reward_ds = ds_norm if ds_norm in DATASOURCE_TO_REWARD else DEFAULT_DATASOURCE
        grouped[reward_ds]["indices"].append(i)
        grouped[reward_ds]["queries"].append(q)
        grouped[reward_ds]["prompts"].append(p)
        grouped[reward_ds]["labels"].append(lbl)

    # Initialize results
    all_rewards = [0.0] * batch_size
    all_scores = [0.0] * batch_size
    all_extra_logs: Dict[str, List[float]] = defaultdict(lambda: [0.0] * batch_size)

    kwargs = dict(kwargs)
    if any(DATASOURCE_TO_REWARD[ds_name]["needs_code_eval"] for ds_name in grouped):
        kwargs.setdefault("code_eval_metric", _get_code_eval_metric())

    # --------- Step 2: Call reward per datasource ---------
    for ds_name, g in grouped.items():
        indices = g["indices"]
        entry = DATASOURCE_TO_REWARD[ds_name]
        reward_fn = entry["resolver"]()

        try:
            result = reward_fn(queries=g["queries"], prompts=g["prompts"], labels=g["labels"], **kwargs)
            rewards = result.get("rewards", torch.zeros(len(indices)))
            scores = result.get("scores", rewards)
            extra_logs = result.get("extra_logs", {})

            if isinstance(rewards, torch.Tensor):
                rewards = rewards.tolist()
            if isinstance(scores, torch.Tensor):
                scores = scores.tolist()

            for j, idx in enumerate(indices):
                all_rewards[idx] = float(rewards[j])
                all_scores[idx] = float(scores[j])

            for key, values in extra_logs.items():
                if isinstance(values, torch.Tensor):
                    values = values.tolist()
                for j, idx in enumerate(indices):
                    all_extra_logs[key][idx] = float(values[j])

        except Exception as e:
            print(f"[Error] Failed to compute reward for datasource '{ds_name}': {e}")
            traceback.print_exc()

    # --------- Step 3: Build return values ---------
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    extra_logs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in all_extra_logs.items()}

    # --------- Step 4: Datasource statistics ---------
    for ds_name in DATASOURCE_TO_REWARD:
        ratio = [0.0] * batch_size
        pass_rate = [0.0] * batch_size
        reward_sum = [0.0] * batch_size

        if ds_name in grouped:
            indices = grouped[ds_name]["indices"]
            scale = batch_size / len(indices) if indices else 0.0
            for idx in indices:
                ratio[idx] = 1.0
                pass_rate[idx] = all_rewards[idx] * scale
                reward_sum[idx] = all_rewards[idx]

        extra_logs_t[f"ratio_{ds_name}"] = torch.tensor(ratio, dtype=torch.float32)
        extra_logs_t[f"pass_rate_{ds_name}"] = torch.tensor(pass_rate, dtype=torch.float32)
        extra_logs_t[f"reward_sum_{ds_name}"] = torch.tensor(reward_sum, dtype=torch.float32)

    # --------- Step 5: Teacher log-probs (KD) ---------
    _zero = torch.zeros(1, dtype=torch.float32)
    kd_mask = [0.0] * batch_size
    kd_disabled = _kd_mgr.is_kd_global_disabled

    if kd_disabled:
        teacher_log_probs_list = [_zero] * batch_size
        guide_applied = [0.0] * batch_size
        guide_fallback = [0.0] * batch_size
        kd_valid = [0.0] * batch_size
    else:
        kd_indices = [
            i for i, ds in enumerate(normalized_datasources)
            if _kd_mgr.is_kd_enabled(ds)
        ]

        if kd_indices:
            kd_queries = [queries[i] for i in kd_indices]
            kd_prompts = [prompts[i] for i in kd_indices]
            kd_labels = [labels[i] for i in kd_indices]
            kd_token_ids = [query_token_ids[i] for i in kd_indices] if query_token_ids else None
            kd_prompt_token_lens = [prompt_token_lens[i] for i in kd_indices] if prompt_token_lens else None
            kd_datasources = [normalized_datasources[i] for i in kd_indices]

            kd_tlps, kd_ga, kd_gf, kd_valid_part = _kd_mgr.compute_teacher_log_probs(
                kd_queries,
                prompts=kd_prompts,
                labels=kd_labels,
                query_token_ids=kd_token_ids,
                prompt_token_lens=kd_prompt_token_lens,
                datasources=kd_datasources,
            )

            teacher_log_probs_list = [_zero] * batch_size
            guide_applied = [0.0] * batch_size
            guide_fallback = [0.0] * batch_size
            kd_valid = [0.0] * batch_size
            for j, idx in enumerate(kd_indices):
                teacher_log_probs_list[idx] = kd_tlps[j]
                guide_applied[idx] = kd_ga[j]
                guide_fallback[idx] = kd_gf[j]
                kd_valid[idx] = kd_valid_part[j]
                kd_mask[idx] = kd_valid_part[j]
        else:
            teacher_log_probs_list = [_zero] * batch_size
            guide_applied = [0.0] * batch_size
            guide_fallback = [0.0] * batch_size
            kd_valid = [0.0] * batch_size

    extra_logs_t["kd_mask"] = torch.tensor(kd_mask, dtype=torch.float32)
    extra_logs_t["kd_skip_reward"] = torch.tensor(kd_skip_mask, dtype=torch.float32)
    extra_logs_t["guide_kd_applied"] = torch.tensor(guide_applied, dtype=torch.float32)
    extra_logs_t["guide_kd_fallback"] = torch.tensor(guide_fallback, dtype=torch.float32)
    extra_logs_t["teacher_kd_valid"] = torch.tensor(kd_valid, dtype=torch.float32)

    return {
        "rewards": rewards_t,
        "scores": scores_t,
        "extra_logs": extra_logs_t,
        "teacher_log_probs": teacher_log_probs_list,
    }
