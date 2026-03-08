"""统一 Reward 调度器（API 评测 + KD）。

按 datasource 分发到对应评测逻辑，汇总 rewards/scores/extra_logs，
可选计算 teacher per-token logprobs（支持 guided KD）。

支持的 datasource:
- cruxeval   -> 本地 reward / API
- python     -> 本地 HumanEval reward
- bigcodebench -> /eval/bigcodebench
- cpp/sh/ts/js/java/cs -> /eval/multiple

reward_func 返回:
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
# code_eval metric（进程内单例）
# =====================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_EVAL_METRIC_PATH = os.path.join(_SCRIPT_DIR, "code_reward", "code_eval")
_CODE_EVAL_METRIC = None


def _get_code_eval_metric(metric_path: str = CODE_EVAL_METRIC_PATH):
    """懒加载 code_eval metric。"""
    global _CODE_EVAL_METRIC
    if _CODE_EVAL_METRIC is None:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ.setdefault("HF_HOME", "/tmp/hf")
        os.environ.setdefault("HF_MODULES_CACHE", os.path.join(os.environ["HF_HOME"], "modules"))
        import evaluate
        _CODE_EVAL_METRIC = evaluate.load(metric_path)
    return _CODE_EVAL_METRIC


# =====================================================================
# 延迟加载的本地 reward 模块
# =====================================================================

_REWARD_MODULES: Dict[str, Any] = {}


def _get_lazy_reward_func(name: str):
    """懒加载本地 reward 函数（humaneval / cruxeval）。"""
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
# datasource -> reward 函数映射
# =====================================================================

DATASOURCE_TO_REWARD: Dict[str, Dict[str, Any]] = {
    # 延迟加载的本地 reward
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
    # 多语言 HumanEval
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
# KD 管理器 (在此处集中配置所有 KD 参数)
#
# 训练模式速查:
#   纯 RL (无 KD)        -> kd_datasources="none"
#   全量 KD              -> kd_datasources="all"
#   部分 KD              -> kd_datasources="python,cruxeval"
#   Guided KD (注入答案)  -> guide_kd_datasources="all" 或指定子集
#   纯 KD (跳过 reward)   -> skip_reward_datasources="python"
# =====================================================================

_kd_mgr = TeacherKDManager(
    # --- Teacher API ---
    default_url="http://10.222.17.214:8080/v1/completions",
    default_model="zhanlu",
    timeout=600,
    max_workers=1,  # teacher 并发数, 避免过载

    # --- 多教师分流 (按 datasource 路由到不同教师) ---
    teacher_by_datasource={
        "bigcodebench": {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "cruxeval":     {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "sh":           {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "cs":           {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "cpp":          {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "java":         {"url": "http://10.222.17.214:8080/v1/completions", "model": "zhanlu"},
        "agent_summary": {"url": "http://10.222.55.196:8080/v1/completions", "model": "zhanlu"},
    },

    # --- 路由控制 ---
    kd_datasources="all",                    # 哪些 ds 计算 teacher logprobs
    guide_kd_datasources="agent_summary",    # 哪些 ds 启用 guided KD
    skip_reward_datasources="agent_summary", # 哪些 ds 跳过 reward (纯 KD)

    # --- Guided KD 内容 ---
    guide_prefix="\nHere is a reference solution:\n",
    guide_suffix="",
    tokenizer_path="",  # 留空则回退到 default_model
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
# 统一 reward 入口
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
    """统一 reward 入口。

    流程:
    1. 按 datasource 分组，跳过 skip-reward 的样本
    2. 分组调用 reward 函数
    3. 构建 datasource 统计指标
    4. 计算 teacher log-probs（KD 路由）
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

    # --------- Step 1: 按 datasource 分组 ---------
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

    # 初始化结果
    all_rewards = [0.0] * batch_size
    all_scores = [0.0] * batch_size
    all_extra_logs: Dict[str, List[float]] = defaultdict(lambda: [0.0] * batch_size)

    kwargs = dict(kwargs)
    if any(DATASOURCE_TO_REWARD[ds_name]["needs_code_eval"] for ds_name in grouped):
        kwargs.setdefault("code_eval_metric", _get_code_eval_metric())

    # --------- Step 2: 分 datasource 调用 reward ---------
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

    # --------- Step 3: 返回值构建 ---------
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    extra_logs_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in all_extra_logs.items()}

    # --------- Step 4: datasource 统计指标 ---------
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
