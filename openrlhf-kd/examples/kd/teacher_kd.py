"""Teacher Knowledge Distillation (KD) module.

Provides TeacherKDManager class for managing:
1. Teacher model API calls (supports multi-teacher routing by datasource)
2. Guided KD: inject reference answers into teacher prompt
3. Routing control for KD / Guide KD / Skip-Reward
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import requests
import torch 
from transformers import AutoTokenizer


def _parse_ds_set(raw: str) -> Optional[FrozenSet[str]]:
    """Parse datasource set configuration.

    Format:
    - "all"  -> None (match all)
    - "none" / "" -> frozenset() (empty set, match none)
    - "a,b,c" -> frozenset({"a", "b", "c"})
    """
    raw = raw.strip().lower()
    if raw == "all":
        return None
    if raw in ("none", ""):
        return frozenset()
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


class TeacherKDManager:
    """Manages all KD logic on the Teacher side.

    All parameters can be passed at construction time or overridden via environment variables.
    Configure centrally when instantiating in guide-kd-reward.py entry point.

    Typical training mode configuration examples:
    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ Training Mode                 │ Configuration                            │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ Pure RL (no KD)               │ kd_datasources="none"                    │
    │ Full KD (all ds distilled)    │ kd_datasources="all" (default)           │
    │ Partial KD                    │ kd_datasources="python,cruxeval"         │
    │ Guided KD (inject reference)   │ guide_kd_datasources="all" or subset    │
    │ Pure KD (skip reward eval)     │ skip_reward_datasources="agent_summary"  │
    └──────────────────────────────┴──────────────────────────────────────────┘

    Args:
        default_url: Default teacher vLLM API URL
        default_model: Default teacher model name
        timeout: Teacher API timeout (seconds)
        max_workers: Teacher API concurrency thread count
        teacher_by_datasource: Multi-teacher routing config {ds: {"url": ..., "model": ...}}
        kd_datasources: Which ds compute teacher logprobs ("all"/"none"/comma-separated)
        guide_kd_datasources: Which ds enable guided KD ("all"/"none"/comma-separated)
        skip_reward_datasources: Which ds skip reward evaluation ("none"/comma-separated)
        guide_prefix: Prefix text injected for guided KD
        guide_suffix: Suffix text injected for guided KD
        tokenizer_path: Tokenizer path for guided KD
    """

    def __init__(
        self,
        # --- Teacher API ---
        default_url: str = "http://localhost/v1/completions",
        default_model: str = "zhanlu",
        timeout: int = 600,
        max_workers: int = 1,
        teacher_by_datasource: Optional[Dict[str, Dict[str, str]]] = None,
        # --- Routing control ---
        kd_datasources: str = "all",
        guide_kd_datasources: str = "agent_summary",
        skip_reward_datasources: str = "agent_summary",
        # --- Guided KD ---
        guide_prefix: str = "\nHere is a reference solution:\n",
        guide_suffix: str = "",
        tokenizer_path: str = "",
    ):
        # Teacher API config
        self.default_url = os.getenv("TEACHER_VLLM_COMPLETION_URL", default_url)
        self.default_model = os.getenv("TEACHER_MODEL_NAME", default_model)
        self.timeout = int(os.getenv("TEACHER_TIMEOUT", str(timeout)))
        self.max_workers = int(os.getenv("TEACHER_MAX_WORKERS", str(max_workers)))

        # Multi-teacher routing by datasource (pass dict config directly)
        self.teacher_by_ds: Dict[str, Dict[str, str]] = teacher_by_datasource or {}

        # Routing control: parse ds sets
        self.kd_ds = _parse_ds_set(kd_datasources)
        self.guide_ds = _parse_ds_set(guide_kd_datasources)
        self.skip_reward_ds = _parse_ds_set(skip_reward_datasources) or frozenset()

        # Guided KD config
        self.guide_prefix = os.getenv("GUIDE_KD_PREFIX", guide_prefix)
        self.guide_suffix = os.getenv("GUIDE_KD_SUFFIX", guide_suffix)
        self._tokenizer_path = os.getenv("GUIDE_KD_TOKENIZER_NAME_OR_PATH", tokenizer_path).strip()

        # Cache (lazy load)
        self._tokenizer = None
        self._tokenizer_checked = False
        self._chat_suffix_cached = False  # Distinguish "not probed" vs "probed and empty"
        self._chat_suffix: Optional[str] = None
        self._chat_suffix_ids: Optional[List[int]] = None

    # -----------------------------------------------------------------
    # Routing predicates
    # -----------------------------------------------------------------

    def is_kd_enabled(self, ds: str) -> bool:
        """Whether this datasource should compute teacher logprobs."""
        return self.kd_ds is None or ds in self.kd_ds

    def is_guide_enabled(self, ds: str) -> bool:
        """Whether this datasource has guided KD enabled (inject reference answers)."""
        return self.guide_ds is None or ds in self.guide_ds

    @property
    def is_kd_global_disabled(self) -> bool:
        """Whether KD is globally disabled (i.e. KD_DATASOURCES="none")."""
        return self.kd_ds is not None and len(self.kd_ds) == 0

    def is_skip_reward(self, ds: str) -> bool:
        """Whether this datasource skips reward evaluation (pure KD)."""
        return bool(self.skip_reward_ds) and ds in self.skip_reward_ds

    # -----------------------------------------------------------------
    # Teacher API
    # -----------------------------------------------------------------

    def get_teacher_config(self, ds: Optional[str] = None) -> Tuple[str, str]:
        """Get teacher (url, model) for the given datasource."""
        if ds:
            ds_norm = ds.lower().strip()
            route = self.teacher_by_ds.get(ds_norm, {})
            url, model = route.get("url"), route.get("model")
            if url and model:
                return url, model
        return self.default_url, self.default_model

    def call_api(self, prompt: Union[str, List[int]],
                 ds: Optional[str] = None) -> List[float]:
        """Call teacher vLLM API, return per-token logprobs (next-token aligned)."""
        url, model = self.get_teacher_config(ds)
        payload = {
            "model": model, "prompt": prompt,
            "max_tokens": 0, "temperature": 0,
            "logprobs": 1, "echo": True,
        }
        last_err: Optional[Exception] = None
        for _ in range(2):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout,
                                  proxies={"http": "", "https": ""})
                r.raise_for_status()
                return self._extract_logprobs(r.json())
            except Exception as e:
                last_err = e
                time.sleep(1)
        raise last_err  # type: ignore

    @staticmethod
    def _extract_logprobs(resp: Dict[str, Any]) -> List[float]:
        """Extract token logprobs from vLLM completion response."""
        choices = resp.get("choices") or []
        if not choices:
            return []
        lp_obj = choices[0].get("logprobs") or {}
        lps = lp_obj.get("token_logprobs") or []
        tokens = lp_obj.get("tokens") or []
        # When echo=True, drop first item to align next-token prediction
        if tokens and len(lps) == len(tokens) and lps:
            lps = lps[1:]
        return [float(x) for x in lps if x is not None]

    # -----------------------------------------------------------------
    # Guided KD: Tokenizer & Prompt building
    # -----------------------------------------------------------------

    def get_tokenizer(self):
        """Lazy-load guided KD tokenizer."""
        if not self._tokenizer_checked:
            self._tokenizer_checked = True
            path = self._tokenizer_path or self.default_model
            if not path:
                return None
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=True, use_fast=True)
            except Exception as e:
                print(f"[Warning] Failed to load tokenizer '{path}': {e}")
        return self._tokenizer

    def _get_chat_suffix(self) -> Tuple[Optional[str], Optional[List[int]]]:
        """Probe chat template suffix (tokens between user message end and generation prompt)."""
        if self._chat_suffix_cached:
            return self._chat_suffix, self._chat_suffix_ids

        self._chat_suffix_cached = True
        tk = self.get_tokenizer()
        if not tk:
            return None, None

        sentinel = "<<<SENTINEL>>>"
        try:
            probe = tk.apply_chat_template(
                [{"role": "user", "content": sentinel}],
                tokenize=False, add_generation_prompt=True)
            idx = probe.find(sentinel)
            if idx >= 0:
                self._chat_suffix = probe[idx + len(sentinel):]
                if self._chat_suffix:
                    self._chat_suffix_ids = tk.encode(
                        self._chat_suffix, add_special_tokens=False)
        except Exception:
            pass
        return self._chat_suffix, self._chat_suffix_ids

    def _build_guided_input(
        self, query_ids: List[int], prompt: str, label: str,
        prompt_len_hint: Optional[int] = None,
    ) -> Optional[Tuple[List[int], int, int, int]]:
        """Build token sequence with guide injected.

        Returns:
            (guided_ids, guided_prompt_len, guide_delta_len, original_len)
            Returns None if build fails.
        """
        tk = self.get_tokenizer()
        if not tk:
            return None

        suffix, suffix_ids = self._get_chat_suffix()
        if not suffix:
            return None

        # Determine prompt boundary
        if prompt_len_hint and 0 < prompt_len_hint < len(query_ids):
            p_len = prompt_len_hint
        else:
            p_ids = tk.encode(prompt, add_special_tokens=False)
            if not p_ids:
                return None
            p_len = len(p_ids)
            if p_len > len(query_ids) or query_ids[:p_len] != p_ids:
                # Longest common prefix fallback
                p_len = 0
                for a, b in zip(query_ids, p_ids):
                    if a != b:
                        break
                    p_len += 1
            if p_len <= 0:
                return None

        resp_ids = query_ids[p_len:]
        orig_p_ids = list(query_ids[:p_len])
        guide_text = f"{self.guide_prefix}{label}{self.guide_suffix}"
        if not guide_text.strip():
            return None

        # Prefer: token-level concatenation
        guided_p_ids = None
        if (suffix_ids is not None
                and len(suffix_ids) <= len(orig_p_ids)
                and orig_p_ids[-len(suffix_ids):] == suffix_ids):
            g_ids = tk.encode(guide_text, add_special_tokens=False)
            if g_ids:
                guided_p_ids = orig_p_ids[:-len(suffix_ids)] + g_ids + suffix_ids

        # Fallback: text-level re-encoding
        if guided_p_ids is None:
            # Compatible with single-text-level guide fallback
            prompt_text = prompt
            if suffix:
                if not prompt_text.endswith(suffix):
                    return None
                inject_pos = len(prompt_text) - len(suffix)
                guided_text = prompt_text[:inject_pos] + guide_text + prompt_text[inject_pos:]
            else:
                guided_text = prompt_text + guide_text
            
            guided_p_ids = tk.encode(guided_text, add_special_tokens=False)
            if not guided_p_ids:
                return None

        delta = len(guided_p_ids) - p_len
        if delta <= 0:
            return None
        return guided_p_ids + resp_ids, len(guided_p_ids), delta, len(query_ids)

    @staticmethod
    def _validate_plain_logprobs(lps: List[float], orig_len: int) -> None:
        """Validate teacher output length for standard KD."""
        expected = max(orig_len - 1, 0)
        if len(lps) != expected:
            raise ValueError(f"teacher logprobs length mismatch: {len(lps)} vs {expected}")

    @staticmethod
    def _validate_guided_logprobs(
        lps: List[float], g_p_len: int, delta: int, orig_len: int
    ) -> None:
        """Validate that guided teacher output still covers the full response."""
        orig_p_len = g_p_len - delta
        if orig_p_len < 0 or orig_len < orig_p_len:
            raise ValueError(
                f"invalid guided metadata: g_p_len={g_p_len}, delta={delta}, orig_len={orig_len}"
            )

        expected_resp = orig_len - orig_p_len
        actual_resp = max(len(lps) - max(g_p_len - 1, 0), 0)
        if actual_resp != expected_resp:
            raise ValueError(
                f"guided KD response length mismatch: {actual_resp} vs {expected_resp}"
            )

    @staticmethod
    def _recover_logprobs(lps: List[float], g_p_len: int,
                          delta: int, orig_len: int) -> List[float]:
        """Recover guided logprobs to original sequence length (pad prompt with 0, align response)."""
        target = max(orig_len - 1, 0)
        if target == 0:
            return []
        orig_p_len = g_p_len - delta
        resp_lps = lps[max(g_p_len - 1, 0):]
        kept = [0.0] * max(orig_p_len - 1, 0) + resp_lps
        if len(kept) >= target:
            return kept[:target]
        return kept + [0.0] * (target - len(kept))

    # -----------------------------------------------------------------
    # Batch computation entry
    # -----------------------------------------------------------------

    def compute_teacher_log_probs(
        self,
        queries: List[str],
        prompts: List[str],
        labels: List[str],
        query_token_ids: Optional[List[List[int]]] = None,
        prompt_token_lens: Optional[List[int]] = None,
        datasources: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], List[float], List[float], List[float]]:
        """Batch compute teacher logprobs (supports guided KD and fallback).

        Returns:
            (teacher_logprobs_list, guide_applied, guide_fallback, kd_valid)
            - teacher_logprobs_list: one 1D tensor per sample
            - guide_applied: 1.0 means guided KD was successfully used
            - guide_fallback: 1.0 means fallback to plain KD after guided failure
            - kd_valid: 1.0 means teacher logprobs passed length validation
        """
        n = len(queries)
        if n == 0:
            return [], [], [], []

        _z = torch.zeros(1, dtype=torch.float32)

        # Prepare inputs (prefer token ids)
        has_ids = query_token_ids is not None and len(query_token_ids) == n
        inputs: List[Any] = list(query_token_ids) if has_ids else list(queries)
        orig_inputs = list(inputs)
        metas: List[Optional[Tuple[int, int, int]]] = [None] * n
        applied = [0.0] * n
        fallback = [0.0] * n
        valid = [0.0] * n

        # Build guided inputs
        needs_guide = (
            has_ids
            and prompts is not None
            and labels is not None
            and any(
                self.is_guide_enabled((datasources[i] if datasources else "").lower().strip())
                for i in range(n)
            )
        )
        if needs_guide:
            if self.get_tokenizer() is not None:
                for i in range(n):
                    ds = (datasources[i] if datasources else "").lower().strip()
                    if not self.is_guide_enabled(ds):
                        continue
                    ptl = prompt_token_lens[i] if prompt_token_lens and i < len(prompt_token_lens) else None
                    guided = self._build_guided_input(
                        query_token_ids[i], prompts[i], labels[i], ptl)  # type: ignore
                    if guided:
                        inputs[i] = guided[0]
                        metas[i] = guided[1:]  # type: ignore
                        applied[i] = 1.0

        # Concurrent teacher API calls
        results: List[Optional[torch.Tensor]] = [None] * n
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            f2i = {
                ex.submit(self.call_api, inputs[i],
                          datasources[i] if datasources else None): i
                for i in range(n)
            }
            for fut in as_completed(f2i):
                idx = f2i[fut]
                try:
                    lps = fut.result()
                    if metas[idx]:
                        self._validate_guided_logprobs(lps, *metas[idx])
                        lps = self._recover_logprobs(lps, *metas[idx])
                    elif has_ids:
                        self._validate_plain_logprobs(lps, len(query_token_ids[idx]))  # type: ignore[arg-type]
                    results[idx] = torch.tensor(lps, dtype=torch.float32) if lps else _z
                    valid[idx] = 1.0
                except Exception as e:
                    # Guided failed -> fallback to plain KD
                    if applied[idx] > 0:
                        applied[idx] = 0.0
                        fallback[idx] = 1.0
                        try:
                            ds = datasources[idx] if datasources else None
                            lps = self.call_api(orig_inputs[idx], ds)
                            if has_ids:
                                self._validate_plain_logprobs(lps, len(query_token_ids[idx]))  # type: ignore[arg-type]
                            results[idx] = torch.tensor(lps, dtype=torch.float32) if lps else _z
                            valid[idx] = 1.0
                            continue
                        except Exception as e2:
                            print(f"[Warning] Guided KD fallback failed at {idx}: {e2}")
                    print(f"[Warning] Teacher API failed at {idx}: {e}")
                    results[idx] = _z

        return [r if r is not None else _z for r in results], applied, fallback, valid
