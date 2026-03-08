"""Remote API evaluation module.

Supported evaluation types:
- BigCodeBench: /eval/bigcodebench
- Multi-language HumanEval: /eval/multiple (cpp/sh/java/js/ts/cs)

Each datasource's reward function returns a unified format:
    {"rewards": Tensor, "scores": Tensor, "extra_logs": {}}
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests
import torch


# =====================================================================
# API Configuration
# =====================================================================

BIGCODE_URL = "http:"      # BigCodeBench evaluation service
MULTIPLE_URL = "http:"     # Multi-language evaluation service

HEADERS = {"Content-Type": "application/json"}
PROXIES: Dict[str, str] = {"http": "", "https": ""}

# Concurrency
MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "8"))

# Timeout (seconds)
TIMEOUT_MULTIPLE = int(os.getenv("EVAL_TIMEOUT_MULTIPLE", "120"))
TIMEOUT_BIGCODEBENCH = int(os.getenv("EVAL_TIMEOUT_BIGCODEBENCH", "180"))

# Retry
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "1"))


# =====================================================================
# Code Extraction (Post-processing)
# =====================================================================

_MULTI_LANG_SUFFIXES = (
    "python|go|rs|rust|ts|scala|d|r|php|csharp|bash|javascript|"
    "cpp|cs|java|typescript|js|sh|julia|swift|perl|lua|rkt|racket|rb|ruby"
)


def extract_code_block_multiple(code: str) -> str:
    """Extract multi-language code from markdown block; returns last matched block."""
    pattern = rf"```(?:{_MULTI_LANG_SUFFIXES})(.*?)```"
    matches = re.findall(pattern, code, re.DOTALL)
    return matches[-1].strip() if matches else code.strip()


def extract_code_block_bigcode(code: str) -> str:
    """Extract BigCodeBench code block; prefer block containing `def`."""
    matches = re.findall(r"```(?:python\s*)?([\s\S]*?)```", code, re.DOTALL)
    if not matches:
        return "none"
    for block in reversed(matches):
        if "def" in block:
            return block.strip()
    return matches[-1].strip()

# =====================================================================
# HTTP Request
# =====================================================================

def _request_with_retry(url: str, payload: Dict, timeout: int) -> Dict[str, Any]:
    """HTTP POST with retry; retries only on Timeout or ConnectionError."""
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=HEADERS,
                                 timeout=timeout, proxies=PROXIES)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            raise
    raise last_error  # type: ignore


# =====================================================================
# Single API Calls (extend with your custom API services)
# =====================================================================

def call_eval_bigcodebench(solution: str, test: str) -> Dict[str, Any]:
    """Call BigCodeBench evaluation API."""
    return _request_with_retry(
        f"{BIGCODE_URL}/eval/bigcodebench",
        {"solution": solution, "test": test},
        TIMEOUT_BIGCODEBENCH,
    )


def call_eval_multiple(solution: str, test: str, language: str = "python") -> Dict[str, Any]:
    """Call multi-language HumanEval evaluation API."""
    return _request_with_retry(
        f"{MULTIPLE_URL}/eval/multiple",
        {"solution": solution, "test": test, "language": language},
        TIMEOUT_MULTIPLE,
    )


# =====================================================================
# Batch Concurrency
# =====================================================================

def _batch_call(fn: Callable, args_list: List[tuple],
                max_workers: int = MAX_WORKERS) -> List[Dict[str, Any]]:
    """
    Execute batch calls concurrently while preserving output order.
    On exception, returns {success: False, error: ...} at that index.
    """
    n = len(args_list)
    if n == 0:
        return []

    results: List[Optional[Dict[str, Any]]] = [None] * n
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(fn, *args): i for i, args in enumerate(args_list)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                error_count += 1
                results[idx] = {"success": False, "error": f"{type(e).__name__}: {e}"}

    if error_count:
        print(f"[ERROR API] {error_count}/{n} requests failed")

    return results  # type: ignore


def _api_results_to_reward(api_res: List[Dict]) -> Dict[str, Any]:
    """Convert API result list to unified reward format."""
    rewards = torch.tensor(
        [1.0 if r.get("success") else 0.0 for r in api_res],
        dtype=torch.float32,
    )
    return {"rewards": rewards, "scores": rewards, "extra_logs": {}}


# =====================================================================
# Reward Functions per Datasource
# =====================================================================

def reward_bigcodebench_api(queries: List[str], prompts: List[str],
                            labels: List[str], **kwargs) -> Dict[str, Any]:
    """BigCodeBench API reward."""
    processed = [extract_code_block_bigcode(q) for q in queries]
    api_res = _batch_call(call_eval_bigcodebench, list(zip(processed, labels)))
    return _api_results_to_reward(api_res)


def make_multiple_reward(language: str) -> Callable:
    """Factory: create reward function for specified language (/eval/multiple)."""
    def _reward(queries: List[str], prompts: List[str],
                labels: List[str], **kwargs) -> Dict[str, Any]:
        processed = [extract_code_block_multiple(q) for q in queries]
        args_list = [(code, test, language) for code, test in zip(processed, labels)]
        api_res = _batch_call(call_eval_multiple, args_list)
        return _api_results_to_reward(api_res)
    return _reward
