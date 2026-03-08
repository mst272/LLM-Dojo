# Reward / KD / Guide-KD

[English](README.md) | 中文版

支持 `RLVR`、`KD` 和 `Guide-KD` 的混合训练。

- 通过 `datasource` 路由 reward、KD、Guide-KD 和 teacher
- 支持纯 RLVR、纯 KD、纯 Guide-KD、完全混合及部分混合

## 目录

- [数据格式](#数据格式)
- [核心配置](#核心配置)
- [模式](#模式)
- [快速开始](#快速开始)
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

- `label` 同时也是 Guide-KD 的参考答案
- `datasource` 为数据类型；后续配置按数据类型分发

## 核心配置

核心配置位于 `examples/guide-kd-reward.py` 中的 `TeacherKDManager(...)`。

| 字段 | 说明 |
| --- | --- |
| `default_url` / `default_model` | 默认 teacher URL |
| `teacher_by_datasource` | 多 teacher 路由 |
| `kd_datasources` | 参与 KD 的 datasource |
| `guide_kd_datasources` | 启用 Guide-KD 的 datasource |
| `skip_reward_datasources` | 跳过 reward、仅使用 KD 的 datasource |
| `guide_prefix` / `guide_suffix` | Guide 注入模板 |
| `tokenizer_path` | Guide-KD 使用的 tokenizer |

Datasource 字段支持 `all`、`none` 或逗号分隔的列表。

### `DATASOURCE_TO_REWARD`

`DATASOURCE_TO_REWARD` 位于 `examples/guide-kd-reward.py`，将 `datasource` 映射到具体的 reward 实现。

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
        "resolver": reward_bigcodebench_api,
        "needs_code_eval": False,
    },
}
```

- `resolver`：返回实际的 reward 函数；可连接本地 reward 或 API reward
- `needs_code_eval`：是否注入 `code_eval` 指标；当前由本地 reward 使用

添加新 datasource 时，先在此注册，再决定是否加入 `kd_datasources` / `guide_kd_datasources`。

### 多 Teacher

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

若 datasource 不在 `teacher_by_datasource` 中，则回退到 `default_url` / `default_model`。

### 当前示例 Reward 环境

| 类型 | datasource | 入口 | 说明 |
| --- | --- | --- | --- |
| 本地 Reward | `python`、`cruxeval` | `examples/code_reward/local/*.py` | 本地执行评估，复用 `code_eval` 指标 |
| API Reward | `bigcodebench` | `examples/code_reward/api/api_eval.py` | 调用远程评估服务，返回统一 reward；需自行构建 API URL |
| Teacher / KD | `kd_datasources` 中的 datasource | `examples/kd/teacher_kd.py` | 调用 teacher 补全 API，支持多 teacher 和 Guide-KD |

### 如何扩展

1. 添加本地 reward：在 `examples/code_reward/local/` 下创建 `*_reward.py`，实现 `reward_func(queries, prompts, labels, **kwargs)`。
2. 添加 API reward：在 `examples/code_reward/api/api_eval.py` 中添加新的 API 调用函数及对应 reward 封装。
3. 注册 datasource：在 `examples/guide-kd-reward.py` 的 `DATASOURCE_TO_REWARD` 中添加新 datasource 映射。
4. 添加 KD / Guide-KD：将 datasource 加入 `kd_datasources`；若需 guided KD，则加入 `guide_kd_datasources`。

## 模式

| 模式 | `kd_datasources` | `guide_kd_datasources` | `skip_reward_datasources` | Trainer 参数 | 效果 |
| --- | --- | --- | --- | --- | --- |
| 纯 RLVR | `none` | `none` | `none` | `--advantage_estimator reinforce/gspo/...` | 仅 reward |
| 纯 KD | `all` 或子集 | `none` | 可选 | `--advantage_estimator kd` | 仅 KD |
| 纯 Guide-KD | 目标子集 | 同子集 | 可选 | `--advantage_estimator kd` | Guide KD |
| 完全混合 | `all` | `none` 或子集 | `none` | `--advantage_estimator reinforce --kd_coef < 1` | 所有 KD datasource 做 RLVR + KD |
| 部分混合 | 目标子集 | `none` 或子集 | `none` | `--advantage_estimator reinforce --kd_coef < 1` | 仅指定 datasource 混合，其余保持 RLVR |
| Guide-KD 混合 | 目标子集 | KD 的子集 | 可选 | `--advantage_estimator reinforce --kd_coef < 1` 或 `--advantage_estimator kd` | 部分 datasource 用 Guide-KD，其余用普通 KD 或 RLVR |

`guide_kd_datasources` 应配置为 `kd_datasources` 的子集。

### `kd_coef` 说明

`kd_coef` 仅在 `--advantage_estimator != kd` 时生效。

```text
kd_adv = teacher_log_probs - student_log_probs
final_adv = (1 - kd_coef) * rlvr_adv + kd_coef * kd_adv
```

- `kd_coef = 0.0`：纯 RLVR
- `0 < kd_coef < 1`：RLVR + KD 混合
- `kd_coef = 1.0`：对启用 KD 的 datasource，等效于纯 KD
- 对于 `skip_reward_datasources` 中的 datasource：即使整体为混合，也按纯 KD 处理，即 final_adv = kd_adv

若 datasource 不在 `kd_datasources` 中：

- 当 `--advantage_estimator != kd`：始终为纯 RLVR
- 当 `--advantage_estimator == kd`：仍会进入 KD 估计器路径，因此纯 KD 训练应将所有活跃 datasource 包含在 `kd_datasources` 中

## 快速开始

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

以下示例仅展示各模式相关的最小配置。

- 实际启用 Guide-KD 的集合 = `kd_datasources ∩ guide_kd_datasources`
- 实际按纯 KD 处理的集合 = `kd_datasources ∩ skip_reward_datasources`
- 在 `--advantage_estimator reinforce`（任意 RL）下，不在 `kd_datasources` 中的 datasource 始终为纯 RLVR

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

- 所有 datasource 仅使用 reward

### 2. 纯 KD，保留 Reward

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

- 所有 datasource 计算 teacher `logprobs`
- Reward 仍可计算用于日志或过滤
- 训练优势仅来自 KD

### 3. 纯 KD，跳过 Reward

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

- 不计算 reward；仅使用 teacher `logprobs`

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

- Teacher prompt 会注入 `label`
- 当 guided 构建失败时自动回退到普通 KD
- 不计算 reward

### 5. 完全 RLVR + KD 混合

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

- 所有 datasource 做 `0.7 * RLVR + 0.3 * KD`

### 6. 部分 Datasource 混合，其余保持 RLVR

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

- `python`、`cruxeval` 使用 `RLVR + KD`
- 不在 `kd_datasources` 中的 datasource 保持纯 RLVR

### 7. 部分 Datasource Guide-KD，其余普通 KD

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

- `summary` 使用 Guide-KD
- `python`、`cruxeval` 使用普通 KD
- 同一训练轮次可混合 guided 与非 guided KD

### 8. Guide-KD + KD + RLVR 全混合

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
- `summary`：纯 Guide-KD
- 不在 `kd_datasources` 中的 datasource：纯 RLVR

### 9. 完全 Guide-KD 混合

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

- 所有 datasource 先做 guided teach，再与 RLVR 混合

### 10. 多 Teacher + 部分混合

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

- `python`：使用 teacher-a 的 `RLVR + KD`
- `summary`：使用 teacher-c 的 Guide-KD
- `cruxeval`：虽配置了专用 teacher，但因不在 `kd_datasources` 中，保持纯 RLVR
