# Reward / KD / Guide-KD

支持 `RLVR`、`KD`、`Guide-KD` 的混合训练。

- 按 `datasource` 路由 reward、KD、Guide-KD 和 teacher
- 支持纯 RLVR、纯 KD、纯 Guide-KD、全量混合和部分混合

## 目录

- [数据格式](#数据格式)
- [核心配置](#核心配置)
- [模式](#模式)
- [Quick Start](#quick-start)
- [配置示例](#配置示例)
- [诊断指标](#诊断指标)


## 数据格式

```json
{
  "input": "...",
  "label": "...",
  "datasource": "python"
}
```

- `label` 对 Guide-KD 同时也是参考答案
- `datasource` 数据类型，后续配置按数据类型分发

## 核心配置

核心配置位于 `examples/guide-kd-reward.py` 中的 `TeacherKDManager(...)`。

| 字段 | 说明 |
| --- | --- |
| `default_url` / `default_model` | 默认 teacher url |
| `teacher_by_datasource` | 多教师路由 |
| `kd_datasources` | 参与 KD 的 datasource |
| `guide_kd_datasources` | 启用 Guide-KD 的 datasource |
| `skip_reward_datasources` | 跳过 reward、只走 KD 的 datasource |
| `guide_prefix` / `guide_suffix` | guide 注入模板 |
| `tokenizer_path` | Guide-KD 使用的 tokenizer |

关于 datasource 字段都支持 `all`、`none` 或逗号分隔列表。

### `DATASOURCE_TO_REWARD`

`DATASOURCE_TO_REWARD` 位于 `examples/guide-kd-reward.py`，负责把 `datasource` 映射到具体 reward 实现。

```python
DATASOURCE_TO_REWARD = {
    "python": {
        "resolver": lambda: _get_lazy_reward_func("humaneval"),
        "needs_code_eval": True,
    },
    "cruxeval": {
        "resolver": lambda: _get_lazy_reward_func("cruxeval"),
        "needs_code_eval": True,
    },
    "bigcodebench": {
        "resolver": lambda: reward_bigcodebench_api,
        "needs_code_eval": False,
    },
}
```

- `resolver`：返回实际 reward 函数，既可以接 local reward，也可以接 API reward
- `needs_code_eval`：是否需要注入 `code_eval` metric，当前 local reward 会用到

新增 datasource 时，通常先在这里注册，再决定是否把它加入 `kd_datasources` / `guide_kd_datasources`。

### 多教师

```python
_kd_mgr = TeacherKDManager(
    default_url="http://teacher-a/v1/completions",
    default_model="teacher-a",
    teacher_by_datasource={
        "python": {"url": "http://teacher-a/v1/completions", "model": "teacher-a"},
        "cruxeval": {"url": "http://teacher-b/v1/completions", "model": "teacher-b"},
        "agent_summary": {"url": "http://teacher-c/v1/completions", "model": "teacher-c"},
    },
)
```

如果某个 datasource 不在 `teacher_by_datasource` 中，会回退到 `default_url` / `default_model`。

### 当前示例reward环境

| 类型 | datasource | 入口 | 说明 |
| --- | --- | --- | --- |
| Local Reward | `python`、`cruxeval` | `examples/code_reward/local/*.py` | 本地执行评测，复用 `code_eval` metric |
| API Reward | `bigcodebench` | `examples/code_reward/api/api_eval.py` | 调用远端评测服务，返回统一 reward 结果，需自行构建API地址 |
| Teacher / KD | `kd_datasources` 中的 datasource | `examples/kd/teacher_kd.py` | 调用 teacher completion API，支持多教师和 Guide-KD |

### 如何扩展

1. 新增 local reward：在 `examples/code_reward/local/` 下新增 `*_reward.py`，实现统一的 `reward_func(queries, prompts, labels, **kwargs)`。
2. 新增 API reward：在 `examples/code_reward/api/api_eval.py` 中增加新的 API 调用函数和对应 reward 包装函数。
3. 注册 datasource：在 `examples/guide-kd-reward.py` 的 `DATASOURCE_TO_REWARD` 中加入新的 datasource 映射。
4. 接入 KD / Guide-KD：把 datasource 加入 `kd_datasources`，如需 guided kd 再加入 `guide_kd_datasources`。

## 模式

| 模式 | `kd_datasources` | `guide_kd_datasources` | `skip_reward_datasources` | Trainer Args | 效果 |
| --- | --- | --- | --- | --- | --- |
| 纯 RLVR | `none` | `none` | `none` | `--advantage_estimator reinforce/gspo/...` | 只用 reward |
| 纯 KD | `all` 或子集 | `none` | 可选 | `--advantage_estimator kd` | 只用 KD |
| 纯 Guide-KD | 目标子集 | 同一子集 | 可选 | `--advantage_estimator kd` | guide KD |
| 全量混合 | `all` | `none` 或子集 | `none` | `--advantage_estimator reinforce --kd_coef < 1` | 所有 KD datasource 都做 RLVR + KD |
| 部分混合 | 目标子集 | `none` 或子集 | `none` | `--advantage_estimator reinforce --kd_coef < 1` | 只有指定 datasource 混合，其余保持 RLVR |
| Guide-KD 混合 | 目标子集 | KD 子集中的一部分 | 可选 | `--advantage_estimator reinforce --kd_coef < 1` 或 `--advantage_estimator kd` | 一部分 datasource 用 Guide-KD，其他用普通 KD 或 RLVR |

guide_kd_datasources 应配置为 kd_datasources 的子集

### `kd_coef`说明

`kd_coef` 只在 `--advantage_estimator != kd` 时生效。

```text
kd_adv = teacher_log_probs - student_log_probs
final_adv = (1 - kd_coef) * rlvr_adv + kd_coef * kd_adv
```

- `kd_coef = 0.0`：纯 RLVR
- `0 < kd_coef < 1`：RLVR + KD 混合
- `kd_coef = 1.0`：对启用 KD 的 datasource 等价于 pure KD
- 在 `skip_reward_datasources` 中的 datasource：即使整体是 hybrid，也会直接按 pure KD， 即 final_adv = kd_adv

如果某个 datasource 不在 `kd_datasources` 中：

- 当 `--advantage_estimator != kd` 时：它始终是纯 RLVR
- 当 `--advantage_estimator == kd` 时：它仍会进入 KD estimator 路径，因此纯 KD 训练应让活跃 datasource 都包含在 `kd_datasources` 中

## Quick Start

```bash
python -m openrlhf.cli.train_ppo_ray \
  --pretrain /path/to/model \
  --prompt_data /path/to/dataset \
  --input_key input \
  --label_key label \
  --remote_rm_url examples/guide-kd-reward.py \
  --advantage_estimator reinforce
```


## 配置示例

下面的示例都只展示和模式相关的最小配置。

- 实际启用 Guide-KD 的集合 = `kd_datasources ∩ guide_kd_datasources`
- 实际按 pure KD 处理的集合 = `kd_datasources ∩ skip_reward_datasources`
- 在 `--advantage_estimator reinforce（任意RL）` 下，不在 `kd_datasources` 中的 datasource 始终是纯 RLVR

### 1. 纯 RLVR

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="none",
    guide_kd_datasources="none",
    skip_reward_datasources="none",
)
```

```bash
--advantage_estimator reinforce
```

说明：

- 所有 datasource 都只走 reward

### 2. 纯 KD，保留 reward

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="all",
    guide_kd_datasources="none",
    skip_reward_datasources="none",
)
```

```bash
--advantage_estimator kd
```

说明：

- 所有 datasource 都计算 teacher `logprobs`
- reward 仍可计算，用于日志或筛选
- 训练优势只来自 KD

### 3. 纯 KD，跳过 reward

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="summary",
    guide_kd_datasources="none",
    skip_reward_datasources="summary",
)
```

```bash
--advantage_estimator kd
```

说明：

- reward不计算，只用 teacher `logprobs`

### 4. 纯 Guide-KD

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="summary",
    guide_kd_datasources="summary",
    skip_reward_datasources="summary",
)
```

```bash
--advantage_estimator kd
```

说明：

- teacher prompt 会注入 `label`
- guided 构造失败时会自动回退到普通 KD
- reward不计算

### 5. 全量 RLVR + KD 混合

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="all",
    guide_kd_datasources="none",
    skip_reward_datasources="none",
)
```

```bash
--advantage_estimator reinforce \
--kd_coef 0.3
```

说明：

- 所有 datasource 都做 `0.7 * RLVR + 0.3 * KD`

### 6. 部分 datasource 混合，其余保持 RLVR

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="python,cruxeval",
    guide_kd_datasources="none",
    skip_reward_datasources="none",
)
```

```bash
--advantage_estimator reinforce \
--kd_coef 0.3
```

说明：

- `python`、`cruxeval` 走 `RLVR + KD`
- 不在 `kd_datasources` 中的 datasource 仍然是纯 RLVR

### 7. 部分 datasource 用 Guide-KD，其余用普通 KD

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="python,cruxeval,summary",
    guide_kd_datasources="summary",
    skip_reward_datasources="summary",
)
```

```bash
--advantage_estimator kd
```

说明：

- `summary` 走 Guide-KD
- `python`、`cruxeval` 走普通 KD
- 同一轮训练中可以混合 guided 和 non-guided KD

### 8. Guide-KD + KD + RLVR 全部混合

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="python,cruxeval,summary",
    guide_kd_datasources="summary",
    skip_reward_datasources="summary",
)
```

```bash
--advantage_estimator reinforce \
--kd_coef 0.3
```

说明：

- `python`、`cruxeval`：`RLVR + KD`
- `summary`：pure Guide-KD
- 不在 `kd_datasources` 中的 datasource：纯 RLVR

### 9. 全量 Guide-KD 混合

```python
_kd_mgr = TeacherKDManager(
    kd_datasources="all",
    guide_kd_datasources="all",
    skip_reward_datasources="none",
)
```

```bash
--advantage_estimator reinforce \
--kd_coef 0.5
```

说明：

- 所有 datasource 都先做 guided teach，再与 RLVR 混合

### 10. 多教师 + 部分混合

```python
_kd_mgr = TeacherKDManager(
    default_url="http://teacher-a/v1/completions",
    default_model="teacher-a",
    teacher_by_datasource={
        "python": {"url": "http://teacher-a/v1/completions", "model": "teacher-a"},
        "cruxeval": {"url": "http://teacher-b/v1/completions", "model": "teacher-b"},
        "summary": {"url": "http://teacher-c/v1/completions", "model": "teacher-c"},
    },
    kd_datasources="python,summary",
    guide_kd_datasources="summary",
    skip_reward_datasources="summary",
)
```

```bash
--advantage_estimator reinforce \
--kd_coef 0.2
```

说明：

- `python`：走 teacher-a 的 `RLVR + KD`
- `summary`：走 teacher-c 的 Guide-KD
- `cruxeval`：虽然配置了专属 teacher，但因为不在 `kd_datasources` 中，仍然是纯 RLVR

