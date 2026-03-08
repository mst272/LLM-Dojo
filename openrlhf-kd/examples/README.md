# Reward / KD / Guide-KD

[中文版](README_zh.md) | English

Supports mixed training of `RLVR`, `KD`, and `Guide-KD`.

- Route reward, KD, Guide-KD and teacher by `datasource`
- Supports pure RLVR, pure KD, pure Guide-KD, full mixed and partial mixed

## Table of Contents

- [Data Format](#data-format)
- [Core Configuration](#core-configuration)
- [Modes](#modes)
- [Quick Start](#quick-start)
- [Configuration Examples](#configuration-examples)
- [Diagnostic Metrics](#diagnostic-metrics)


## Data Format

```json
{
  "input": "...",
  "label": "...",
  "datasource": "python"
}
```

- `label` is also the reference answer for Guide-KD
- `datasource` is the data type; subsequent config dispatches by data type

## Core Configuration

Core configuration is in `TeacherKDManager(...)` in `examples/guide-kd-reward.py`.

| Field | Description |
| --- | --- |
| `default_url` / `default_model` | Default teacher URL |
| `teacher_by_datasource` | Multi-teacher routing |
| `kd_datasources` | Datasources participating in KD |
| `guide_kd_datasources` | Datasources with Guide-KD enabled |
| `skip_reward_datasources` | Datasources that skip reward and only use KD |
| `guide_prefix` / `guide_suffix` | Guide injection template |
| `tokenizer_path` | Tokenizer used for Guide-KD |

Datasource fields support `all`, `none` or comma-separated lists.

### `DATASOURCE_TO_REWARD`

`DATASOURCE_TO_REWARD` is in `examples/guide-kd-reward.py` and maps `datasource` to concrete reward implementations.

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

- `resolver`: returns the actual reward function; can connect to local reward or API reward
- `needs_code_eval`: whether to inject `code_eval` metric; currently used by local reward

When adding a new datasource, register it here first, then decide whether to add it to `kd_datasources` / `guide_kd_datasources`.

### Multi-Teacher

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

If a datasource is not in `teacher_by_datasource`, it falls back to `default_url` / `default_model`.

### Current Example Reward Environments

| Type | datasource | Entry | Description |
| --- | --- | --- | --- |
| Local Reward | `python`, `cruxeval` | `examples/code_reward/local/*.py` | Local execution evaluation, reuses `code_eval` metric |
| API Reward | `bigcodebench` | `examples/code_reward/api/api_eval.py` | Calls remote evaluation service, returns unified reward; build API URL yourself |
| Teacher / KD | datasources in `kd_datasources` | `examples/kd/teacher_kd.py` | Calls teacher completion API, supports multi-teacher and Guide-KD |

### How to Extend

1. Add local reward: create `*_reward.py` under `examples/code_reward/local/`, implement `reward_func(queries, prompts, labels, **kwargs)`.
2. Add API reward: add new API call function and corresponding reward wrapper in `examples/code_reward/api/api_eval.py`.
3. Register datasource: add new datasource mapping in `DATASOURCE_TO_REWARD` in `examples/guide-kd-reward.py`.
4. Add KD / Guide-KD: add datasource to `kd_datasources`; add to `guide_kd_datasources` if guided KD is needed.

## Modes

| Mode | `kd_datasources` | `guide_kd_datasources` | `skip_reward_datasources` | Trainer Args | Effect |
| --- | --- | --- | --- | --- | --- |
| Pure RLVR | `none` | `none` | `none` | `--advantage_estimator reinforce/gspo/...` | Reward only |
| Pure KD | `all` or subset | `none` | optional | `--advantage_estimator kd` | KD only |
| Pure Guide-KD | target subset | same subset | optional | `--advantage_estimator kd` | Guide KD |
| Full mixed | `all` | `none` or subset | `none` | `--advantage_estimator reinforce --kd_coef < 1` | All KD datasources do RLVR + KD |
| Partial mixed | target subset | `none` or subset | `none` | `--advantage_estimator reinforce --kd_coef < 1` | Only specified datasources mixed, others remain RLVR |
| Guide-KD mixed | target subset | subset of KD | optional | `--advantage_estimator reinforce --kd_coef < 1` or `--advantage_estimator kd` | Some datasources use Guide-KD, others use plain KD or RLVR |

guide_kd_datasources should be configured as a subset of kd_datasources

### `kd_coef` Notes

`kd_coef` only takes effect when `--advantage_estimator != kd`.

```text
kd_adv = teacher_log_probs - student_log_probs
final_adv = (1 - kd_coef) * rlvr_adv + kd_coef * kd_adv
```

- `kd_coef = 0.0`: pure RLVR
- `0 < kd_coef < 1`: RLVR + KD mixed
- `kd_coef = 1.0`: for KD-enabled datasources, equivalent to pure KD
- For datasources in `skip_reward_datasources`: even if overall is hybrid, they are treated as pure KD, i.e. final_adv = kd_adv

If a datasource is not in `kd_datasources`:

- When `--advantage_estimator != kd`: it is always pure RLVR
- When `--advantage_estimator == kd`: it still enters the KD estimator path, so pure KD training should include all active datasources in `kd_datasources`

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


## Configuration Examples

The examples below show only the minimal configuration relevant to each mode.

- Actual Guide-KD enabled set = `kd_datasources ∩ guide_kd_datasources`
- Actual pure KD treated set = `kd_datasources ∩ skip_reward_datasources`
- Under `--advantage_estimator reinforce` (any RL), datasources not in `kd_datasources` are always pure RLVR

### 1. Pure RLVR

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

Notes:

- All datasources use reward only

### 2. Pure KD, Keep Reward

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

Notes:

- All datasources compute teacher `logprobs`
- Reward can still be computed for logging or filtering
- Training advantage comes only from KD

### 3. Pure KD, Skip Reward

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

Notes:

- Reward is not computed; only teacher `logprobs` are used

### 4. Pure Guide-KD

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

Notes:

- Teacher prompt will inject `label`
- Automatically falls back to plain KD when guided construction fails
- Reward is not computed

### 5. Full RLVR + KD Mixed

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

Notes:

- All datasources do `0.7 * RLVR + 0.3 * KD`

### 6. Partial Datasource Mixed, Others Remain RLVR

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

Notes:

- `python`, `cruxeval` use `RLVR + KD`
- Datasources not in `kd_datasources` remain pure RLVR

### 7. Partial Datasource Guide-KD, Others Plain KD

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

Notes:

- `summary` uses Guide-KD
- `python`, `cruxeval` use plain KD
- Same training round can mix guided and non-guided KD

### 8. Guide-KD + KD + RLVR All Mixed

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

Notes:

- `python`, `cruxeval`: `RLVR + KD`
- `summary`: pure Guide-KD
- Datasources not in `kd_datasources`: pure RLVR

### 9. Full Guide-KD Mixed

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

Notes:

- All datasources first do guided teach, then mix with RLVR

### 10. Multi-Teacher + Partial Mixed

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

Notes:

- `python`: uses teacher-a's `RLVR + KD`
- `summary`: uses teacher-c's Guide-KD
- `cruxeval`: although configured with dedicated teacher, remains pure RLVR because not in `kd_datasources`
