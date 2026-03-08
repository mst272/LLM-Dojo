"""Teacher 知识蒸馏 (KD) 模块。

提供 TeacherKDManager 类，管理:
1. Teacher 模型 API 调用（支持多教师按 datasource 分流）
2. Guided KD: 在 teacher prompt 中注入参考答案
3. KD / Guide KD / Skip-Reward 的路由控制
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import requests
import torch 
from transformers import AutoTokenizer


def _parse_ds_set(raw: str) -> Optional[FrozenSet[str]]:
    """解析 datasource 集合配置。

    格式:
    - "all"  -> None (表示全部匹配)
    - "none" / "" -> frozenset() (空集合, 不匹配任何)
    - "a,b,c" -> frozenset({"a", "b", "c"})
    """
    raw = raw.strip().lower()
    if raw == "all":
        return None
    if raw in ("none", ""):
        return frozenset()
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


class TeacherKDManager:
    """管理 Teacher 侧的全部 KD 逻辑。

    所有参数均可在构造时传入, 也可通过对应环境变量覆盖默认值。
    在 guide-kd-reward.py 入口处构造实例时集中配置即可。

    典型训练模式配置示例:
    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ 训练模式                     │ 配置                                     │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ 纯 RL (无 KD)                │ kd_datasources="none"                    │
    │ 全量 KD (所有 ds 蒸馏)       │ kd_datasources="all" (默认)              │
    │ 部分 KD                      │ kd_datasources="python,cruxeval"         │
    │ Guided KD (注入参考答案)     │ guide_kd_datasources="all" 或指定子集    │
    │ 纯 KD (跳过 reward 评测)     │ skip_reward_datasources="agent_summary"  │
    └──────────────────────────────┴──────────────────────────────────────────┘

    Args:
        default_url: 默认 teacher vLLM API 地址
        default_model: 默认 teacher 模型名称
        timeout: teacher API 超时 (秒)
        max_workers: teacher API 并发线程数
        teacher_by_datasource: 多教师分流配置 {ds: {"url": ..., "model": ...}}
        kd_datasources: 哪些 ds 计算 teacher logprobs ("all"/"none"/逗号分隔)
        guide_kd_datasources: 哪些 ds 启用 guided KD ("all"/"none"/逗号分隔)
        skip_reward_datasources: 哪些 ds 跳过 reward 评测 ("none"/逗号分隔)
        guide_prefix: guided KD 注入的前缀文本
        guide_suffix: guided KD 注入的后缀文本
        tokenizer_path: guided KD 使用的 tokenizer 路径
    """

    def __init__(
        self,
        # --- Teacher API ---
        default_url: str = "http://localhost/v1/completions",
        default_model: str = "zhanlu",
        timeout: int = 600,
        max_workers: int = 1,
        teacher_by_datasource: Optional[Dict[str, Dict[str, str]]] = None,
        # --- 路由控制 ---
        kd_datasources: str = "all",
        guide_kd_datasources: str = "agent_summary",
        skip_reward_datasources: str = "agent_summary",
        # --- Guided KD ---
        guide_prefix: str = "\nHere is a reference solution:\n",
        guide_suffix: str = "",
        tokenizer_path: str = "",
    ):
        # Teacher API 配置
        self.default_url = os.getenv("TEACHER_VLLM_COMPLETION_URL", default_url)
        self.default_model = os.getenv("TEACHER_MODEL_NAME", default_model)
        self.timeout = int(os.getenv("TEACHER_TIMEOUT", str(timeout)))
        self.max_workers = int(os.getenv("TEACHER_MAX_WORKERS", str(max_workers)))

        # 多教师按 datasource 分流 (直接传字典配置)
        self.teacher_by_ds: Dict[str, Dict[str, str]] = teacher_by_datasource or {}

        # 路由控制: 解析 ds 集合
        self.kd_ds = _parse_ds_set(kd_datasources)
        self.guide_ds = _parse_ds_set(guide_kd_datasources)
        self.skip_reward_ds = _parse_ds_set(skip_reward_datasources) or frozenset()

        # Guided KD 配置
        self.guide_prefix = os.getenv("GUIDE_KD_PREFIX", guide_prefix)
        self.guide_suffix = os.getenv("GUIDE_KD_SUFFIX", guide_suffix)
        self._tokenizer_path = os.getenv("GUIDE_KD_TOKENIZER_NAME_OR_PATH", tokenizer_path).strip()

        # 缓存 (懒加载)
        self._tokenizer = None
        self._tokenizer_checked = False
        self._chat_suffix_cached = False  # 区分 "未探测" 和 "探测后为空"
        self._chat_suffix: Optional[str] = None
        self._chat_suffix_ids: Optional[List[int]] = None

    # -----------------------------------------------------------------
    # 路由判断
    # -----------------------------------------------------------------

    def is_kd_enabled(self, ds: str) -> bool:
        """该 datasource 是否需要计算 teacher logprobs。"""
        return self.kd_ds is None or ds in self.kd_ds

    def is_guide_enabled(self, ds: str) -> bool:
        """该 datasource 是否启用 guided KD (注入参考答案)。"""
        return self.guide_ds is None or ds in self.guide_ds

    @property
    def is_kd_global_disabled(self) -> bool:
        """判断是否全局关闭了 KD (即 KD_DATASOURCES="none")。"""
        return self.kd_ds is not None and len(self.kd_ds) == 0

    def is_skip_reward(self, ds: str) -> bool:
        """该 datasource 是否跳过 reward 评测 (纯 KD)。"""
        return bool(self.skip_reward_ds) and ds in self.skip_reward_ds

    # -----------------------------------------------------------------
    # Teacher API
    # -----------------------------------------------------------------

    def get_teacher_config(self, ds: Optional[str] = None) -> Tuple[str, str]:
        """获取对应 datasource 的 teacher (url, model)。"""
        if ds:
            ds_norm = ds.lower().strip()
            route = self.teacher_by_ds.get(ds_norm, {})
            url, model = route.get("url"), route.get("model")
            if url and model:
                return url, model
        return self.default_url, self.default_model

    def call_api(self, prompt: Union[str, List[int]],
                 ds: Optional[str] = None) -> List[float]:
        """调用 teacher vLLM API, 返回 per-token logprobs (next-token 对齐)。"""
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
        """从 vLLM completion 响应中提取 token logprobs。"""
        choices = resp.get("choices") or []
        if not choices:
            return []
        lp_obj = choices[0].get("logprobs") or {}
        lps = lp_obj.get("token_logprobs") or []
        tokens = lp_obj.get("tokens") or []
        # echo=True 时丢弃首项, 对齐 next-token 预测
        if tokens and len(lps) == len(tokens) and lps:
            lps = lps[1:]
        return [float(x) for x in lps if x is not None]

    # -----------------------------------------------------------------
    # Guided KD: Tokenizer & Prompt 构建
    # -----------------------------------------------------------------

    def get_tokenizer(self):
        """懒加载 guided KD tokenizer。"""
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
        """探测 chat template 后缀 (user 消息结束到 generation prompt 之间的标记)。"""
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
        """构建注入 guide 后的 token 序列。

        Returns:
            (guided_ids, guided_prompt_len, guide_delta_len, original_len)
            构建失败返回 None。
        """
        tk = self.get_tokenizer()
        if not tk:
            return None

        suffix, suffix_ids = self._get_chat_suffix()
        if not suffix:
            return None

        # 确定 prompt 分界线
        if prompt_len_hint and 0 < prompt_len_hint < len(query_ids):
            p_len = prompt_len_hint
        else:
            p_ids = tk.encode(prompt, add_special_tokens=False)
            if not p_ids:
                return None
            p_len = len(p_ids)
            if p_len > len(query_ids) or query_ids[:p_len] != p_ids:
                # 最长公共前缀 fallback
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

        # 优先: token 级拼接
        guided_p_ids = None
        if (suffix_ids is not None
                and len(suffix_ids) <= len(orig_p_ids)
                and orig_p_ids[-len(suffix_ids):] == suffix_ids):
            g_ids = tk.encode(guide_text, add_special_tokens=False)
            if g_ids:
                guided_p_ids = orig_p_ids[:-len(suffix_ids)] + g_ids + suffix_ids

        # 回退: 文本级重编码
        if guided_p_ids is None:
            # 兼容单条文本级别的 guide fallback
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
        """将 guided logprobs 恢复到原始序列长度 (prompt 补 0, response 对齐)。"""
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
    # 批量计算入口
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
        """批量计算 teacher logprobs (支持 guided KD 与回退)。

        Returns:
            (teacher_logprobs_list, guide_applied, guide_fallback, kd_valid)
            - teacher_logprobs_list: 每样本一个 1D tensor
            - guide_applied: 1.0 表示成功使用 guided KD
            - guide_fallback: 1.0 表示 guided 失败后回退到普通 KD
            - kd_valid: 1.0 表示拿到了长度校验通过的 teacher logprobs
        """
        n = len(queries)
        if n == 0:
            return [], [], [], []

        _z = torch.zeros(1, dtype=torch.float32)

        # 准备输入 (优先用 token ids)
        has_ids = query_token_ids is not None and len(query_token_ids) == n
        inputs: List[Any] = list(query_token_ids) if has_ids else list(queries)
        orig_inputs = list(inputs)
        metas: List[Optional[Tuple[int, int, int]]] = [None] * n
        applied = [0.0] * n
        fallback = [0.0] * n
        valid = [0.0] * n

        # 构建 guided 输入
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

        # 并发调用 teacher API
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
                    # guided 失败 -> 回退普通 KD
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
