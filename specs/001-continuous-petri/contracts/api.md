# API Contracts: 连续 Petri 网晶圆加工仿真系统

**Date**: 2026-01-25  
**Feature**: 001-continuous-petri

## Overview

本文档定义了连续 Petri 网晶圆加工仿真系统的公共 API 接口。系统作为强化学习环境，提供标准的初始化、执行和查询接口。

## Core API: Petri Class

### Initialization

#### `__init__(config=None, stop_on_scrap=None, training_phase=None, reward_config=None)`

初始化 Petri 网环境。

**Parameters**:
- `config: Optional[PetriEnvConfig]` - 环境配置对象（推荐使用）
- `stop_on_scrap: Optional[bool]` - 报废时是否停止（如果 config 为 None 时使用）
- `training_phase: Optional[int]` - 训练阶段（如果 config 为 None 时使用）
- `reward_config: Optional[Dict[str, int]]` - 奖励配置（如果 config 为 None 时使用）

**Returns**: `None`

**Behavior**:
- 如果 `config` 为 None，使用默认配置或传入的参数创建 `PetriEnvConfig`
- 构建 Petri 网结构（模块、机器人、路线）
- 初始化所有库所、变迁、初始标记
- 设置系统时间为 1

**Example**:
```python
from solutions.Continuous_model.pn import Petri
from data.petri_configs.env_config import PetriEnvConfig

# 方式1：使用配置对象（推荐）
config = PetriEnvConfig(n_wafer=4, training_phase=2)
env = Petri(config=config)

# 方式2：使用参数
env = Petri(stop_on_scrap=True, training_phase=2)
```

---

### State Management

#### `reset() -> None`

重置环境到初始状态。

**Returns**: `None`

**Behavior**:
- 重置系统时间为 1
- 恢复初始标记 `m = m0`
- 克隆原始库所状态
- 清空变迁日志
- 重置所有内部状态（报废计数、停滞时间、统计信息等）

**Example**:
```python
env.reset()
assert env.time == 1
assert env.m[env.id2p_name.index("LP")] == env.n_wafer
```

---

### Action Execution

#### `step(t=None, wait=False, with_reward=False, detailed_reward=False) -> Tuple`

执行一步动作。

**Parameters**:
- `t: Optional[int]` - 要执行的变迁索引（当 wait=False 时）
- `wait: bool` - 是否执行 WAIT 动作（默认 False）
- `with_reward: bool` - 是否返回奖励（默认 False）
- `detailed_reward: bool` - 是否返回详细奖励分解（仅当 with_reward=True 时有效）

**Returns**:
- 如果 `with_reward=True` 且 `detailed_reward=False`: `(done: bool, reward: float, scrap: bool)`
- 如果 `with_reward=True` 且 `detailed_reward=True`: `(done: bool, reward_dict: Dict[str, float], scrap: bool)`
- 否则: `(done: bool, scrap: bool)`

**Behavior**:
- 如果 `wait=True`: 推进时间 5 秒，计算奖励，检查报废/停滞
- 如果 `wait=False`: 执行变迁 `t`，计算奖励，检查完成/报废
- `done=True` 表示 episode 结束（完成或报废）
- `scrap=True` 表示因报废而结束

**Example**:
```python
# 执行变迁动作
enabled = env.get_enable_t()
if enabled:
    t = enabled[0]
    done, reward, scrap = env.step(t=t, with_reward=True)
    
# 执行 WAIT 动作
done, reward, scrap = env.step(wait=True, with_reward=True)

# 获取详细奖励分解
done, reward_dict, scrap = env.step(t=t, with_reward=True, detailed_reward=True)
# reward_dict = {
#     "total": float,
#     "proc_reward": float,
#     "safe_reward": float,
#     "penalty": float,
#     "warn_penalty": float,
#     "transport_penalty": float,
#     "congestion_penalty": float,
#     "time_cost": float
# }
```

---

### Query Interface

#### `get_enable_t() -> List[int]`

获取当前时刻所有可使能的变迁索引列表。

**Returns**: `List[int]` - 可使能的变迁索引列表

**Behavior**:
- 检查所有变迁的前置条件和容量约束
- 检查最早使能时间是否 <= 当前时间
- 返回满足条件的变迁索引列表

**Example**:
```python
enabled = env.get_enable_t()
print(f"当前有 {len(enabled)} 个变迁可使能")
for t in enabled:
    print(f"  - {env.id2t_name[t]}")
```

---

#### `next_enable_time() -> int`

计算下一个变迁最早可使能的时间。

**Returns**: `int` - 下一个变迁最早可使能的时间

**Behavior**:
- 遍历所有可使能的变迁
- 计算每个变迁的最早使能时间
- 返回最小值

**Example**:
```python
next_time = env.next_enable_time()
print(f"下一个变迁最早可在时间 {next_time} 使能")
```

---

#### `calc_wafer_statistics() -> Dict[str, Any]`

计算晶圆滞留时间统计数据。

**Returns**: `Dict[str, Any]` - 统计信息字典（见 data-model.md）

**Behavior**:
- 遍历所有晶圆的统计记录
- 计算系统滞留时间、各腔室滞留时间、运输位滞留时间
- 返回平均值、最大值、计数等统计信息

**Example**:
```python
stats = env.calc_wafer_statistics()
print(f"平均系统滞留时间: {stats['system_avg']:.2f} 秒")
print(f"已完成晶圆数: {stats['completed_count']}")
for chamber, data in stats['chambers'].items():
    print(f"{chamber}: 平均 {data['avg']:.2f} 秒, 最大 {data['max']} 秒")
```

---

### Visualization

#### `render_gantt(out_path: str) -> None`

生成甘特图可视化。

**Parameters**:
- `out_path: str` - 输出文件路径

**Returns**: `None`

**Behavior**:
- 解析变迁执行日志
- 生成甘特图，展示所有晶圆的加工时间线
- 保存到指定路径

**Example**:
```python
env.render_gantt("results/continuous_gantt.png")
```

---

### Internal Methods (Not Public API)

以下方法为内部实现，不建议直接调用：

- `_fire(t: int) -> None` - 执行变迁（内部方法）
- `_resource_enable() -> np.ndarray` - 检查资源使能（内部方法）
- `_earliest_enable_time(t: int) -> int` - 计算最早使能时间（内部方法）
- `calc_reward(t1: int, t2: int, ...) -> float` - 计算奖励（内部方法，可通过 step 间接使用）
- `_check_scrap() -> bool` - 检查报废（内部方法）
- `_check_idle_timeout() -> bool` - 检查停滞（内部方法）

---

## Configuration API: PetriEnvConfig Class

### Initialization

#### `__init__(**kwargs) -> None`

初始化环境配置。

**Parameters**: 所有配置参数（见 data-model.md）

**Example**:
```python
from data.petri_configs.env_config import PetriEnvConfig

config = PetriEnvConfig(
    n_wafer=4,
    training_phase=2,
    R_done=20,
    R_finish=500,
    R_scrap=100
)
```

### Serialization

#### `save(filepath: str) -> None`

保存配置到 JSON 文件。

**Parameters**:
- `filepath: str` - 输出文件路径

**Example**:
```python
config.save("data/petri_configs/my_config.json")
```

#### `load(filepath: str) -> PetriEnvConfig`

从 JSON 文件加载配置。

**Parameters**:
- `filepath: str` - 配置文件路径

**Returns**: `PetriEnvConfig` - 配置对象

**Example**:
```python
config = PetriEnvConfig.load("data/petri_configs/phase2_config.json")
```

---

## Error Handling

### Invalid Transition

如果尝试执行不可使能的变迁，系统行为未定义（当前实现可能不检查，建议先调用 `get_enable_t()` 验证）。

### Configuration Errors

- 无效的配置参数会导致运行时错误
- 建议使用 `PetriEnvConfig` 类进行配置验证

### Time Overflow

如果系统时间超过 `MAX_TIME`（20000），`step()` 会返回 `done=True, scrap=True`。

---

## Integration with RL Frameworks

系统可以作为强化学习环境使用：

```python
import gym
from stable_baselines3 import PPO

# 包装为 gym 环境（需要额外实现）
env = PetriEnvWrapper(Petri(config=config))

# 训练
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

**Required Interface**:
- `reset() -> observation`
- `step(action) -> (observation, reward, done, info)`
- `observation_space`
- `action_space`

（当前实现需要包装器来适配标准 RL 接口）
