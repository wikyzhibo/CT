# Data Model: 连续 Petri 网晶圆加工仿真系统

**Date**: 2026-01-25  
**Feature**: 001-continuous-petri

## Overview

本文档定义了连续 Petri 网晶圆加工仿真系统的核心数据模型。系统基于 Petri 网理论，使用库所（Place）、令牌（Token）和变迁（Transition）来建模晶圆加工流程。

## Core Entities

### Place (库所)

表示系统中的位置，包括加载端口、加工腔室、运输位、机械手资源等。

**Attributes**:
- `name: str` - 库所名称（如 "LP", "s1", "d_s1"）
- `capacity: int` - 容量（可容纳的最大 token 数）
- `processing_time: int` - 加工时间（秒），对于运输位通常为 0
- `type: int` - 库所类型：
  - `1`: 加工腔室（有驻留约束）
  - `2`: 运输位（delivery place）
  - `3`: 空闲库所（如 LP）
  - `4`: 资源库所（如机械手资源）
  - `5`: 无驻留约束腔室（如 LLC, LLD）
- `tokens: Deque[BasedToken]` - 当前库所中的 token 队列（FIFO）
- `release_schedule: Deque[Tuple[int, int]]` - 释放时间调度队列，格式为 `[(token_id, release_time), ...]`
- `last_machine: int` - 上次分配的机器编号（仅 type=1 使用），用于轮换分配

**Relationships**:
- 一个 Place 可以包含多个 Token（受 capacity 限制）
- 一个 Place 可以连接到多个 Transition（通过前置/后置矩阵）

**Validation Rules**:
- `len(tokens) <= capacity`
- `processing_time >= 0`
- `capacity >= 1`

**State Transitions**:
- Token 进入：`append(token)` → 检查容量 → 更新 `m[place_idx]`
- Token 离开：`pop_head()` → 更新 `m[place_idx]`
- 释放时间更新：`add_release()`, `update_release()`, `pop_release()`

### BasedToken (令牌)

表示晶圆在系统中的实例。

**Attributes**:
- `enter_time: int` - 进入当前库所的时间（系统时间戳）
- `stay_time: int` - 在当前库所的滞留时间（秒）
- `token_id: int` - 晶圆唯一标识符（-1 表示未分配）
- `machine: int` - 分配的机器编号（仅多机器腔室使用，-1 表示未分配）

**Relationships**:
- 一个 Token 只能在一个 Place 中
- 一个 Token 有唯一的 `token_id`，用于追踪整个加工流程

**Validation Rules**:
- `enter_time >= 0`
- `stay_time >= 0`
- `token_id >= -1`（-1 表示资源 token，无晶圆 ID）

**State Transitions**:
- 创建：在变迁执行时创建，继承上游 token 的 `token_id`
- 移动：从源 Place 移除，添加到目标 Place，更新 `enter_time`
- 更新：每次时间推进时更新 `stay_time`

### Transition (变迁)

表示系统中的动作，包括晶圆移动和机械手操作。

**Attributes**:
- `name: str` - 变迁名称（如 "u_LP_s1", "t_s1"）
- `pre: np.ndarray` - 前置矩阵（P × T），表示变迁的前置库所和消耗的 token 数
- `pst: np.ndarray` - 后置矩阵（P × T），表示变迁的后置库所和产生的 token 数
- `ttime: int` - 变迁执行时间（秒），通常为 5 秒

**Relationships**:
- 一个 Transition 有多个前置库所（pre > 0 的位置）
- 一个 Transition 有多个后置库所（pst > 0 的位置）
- 一个 Transition 的执行需要消耗前置库所的 token，并在后置库所产生 token

**Validation Rules**:
- 使能条件：`pre <= m`（前置库所有足够的 token）
- 容量约束：`m + pst <= k`（后置库所不超容量）

**State Transitions**:
- 使能检查：检查前置条件和容量约束
- 执行：消耗前置 token，产生后置 token，推进时间

### Petri (Petri 网)

整个系统模型，包含所有库所、变迁、初始标记、容量约束等。

**Attributes**:
- `pre: np.ndarray` - 前置矩阵（P × T）
- `pst: np.ndarray` - 后置矩阵（P × T）
- `net: np.ndarray` - 净矩阵（pst - pre）
- `m0: np.ndarray` - 初始标记（P,）
- `m: np.ndarray` - 当前标记（P,）
- `k: np.ndarray` - 容量向量（P,）
- `ptime: np.ndarray` - 加工时间向量（P,）
- `ttime: int` - 变迁时间（秒）
- `marks: List[Place]` - 当前库所状态列表
- `ori_marks: List[Place]` - 原始库所状态列表（用于重置）
- `time: int` - 系统当前时间
- `fire_log: List[Dict]` - 变迁执行日志
- `wafer_stats: Dict[int, Dict]` - 晶圆统计信息

**Relationships**:
- 包含多个 Place（通过 `marks` 列表）
- 包含多个 Transition（通过 `pre`, `pst` 矩阵）
- 追踪多个 Token（通过 Place 中的 tokens）

**State Transitions**:
- 初始化：构建 Petri 网结构，设置初始标记
- 执行变迁：`step(t)` → 检查使能 → 执行 `_fire()` → 更新状态
- 等待：`step(wait=True)` → 推进时间 → 计算奖励
- 重置：`reset()` → 恢复到初始状态

## Configuration Entities

### PetriEnvConfig

环境配置类，管理所有可配置参数。

**Attributes**:
- `n_wafer: int` - 晶圆数量
- `c_time: float` - 时间成本系数
- `R_done: float` - 单片完工奖励
- `R_finish: float` - 全部完工奖励
- `R_scrap: float` - 报废惩罚
- `T_warn: int` - 预警阈值（秒）
- `a_warn: float` - 预警惩罚系数
- `T_safe: int` - 安全裕量阈值（秒）
- `b_safe: float` - 安全裕量奖励系数
- `D_Residual_time: int` - 运输位剩余时间
- `P_Residual_time: int` - 加工腔室剩余时间
- `c_release_violation: float` - 释放时间违规惩罚系数
- `idle_timeout: int` - 停滞超时阈值（秒）
- `idle_penalty: float` - 停滞惩罚值
- `stop_on_scrap: bool` - 报废时是否停止
- `training_phase: int` - 训练阶段（1 或 2）
- `reward_config: Dict[str, int]` - 奖励开关配置

**Validation Rules**:
- `n_wafer >= 1`
- `training_phase in [1, 2]`
- 所有时间参数 >= 0
- 所有奖励/惩罚系数合理（避免数值溢出）

## Statistics Entities

### WaferStatistics

晶圆滞留时间统计信息。

**Structure**:
```python
{
    "system_avg": float,      # 平均系统滞留时间
    "system_max": int,        # 最大系统滞留时间
    "system_diff": float,      # 最大最小时间差
    "completed_count": int,    # 已完成晶圆数
    "in_progress_count": int,  # 进行中晶圆数
    "chambers": {             # 各腔室统计
        "PM7/8": {"avg": float, "max": int, "count": int},
        "LLC": {"avg": float, "max": int, "count": int},
        "PM1/2": {"avg": float, "max": int, "count": int},
        "LLD": {"avg": float, "max": int, "count": int},
        "PM9/10": {"avg": float, "max": int, "count": int}
    },
    "transports": {           # 运输位统计（合并）
        "avg": float,
        "max": int,
        "count": int
    },
    "transports_detail": {    # 各运输位详细统计
        "d_s1": {"avg": float, "max": int, "count": int},
        ...
    }
}
```

## Data Flow

### Token Flow

```
LP (初始) → d_s1 → s1 (PM7/8) → d_s2 → s2 (LLC) → d_s3 → 
s3 (PM1/2) → d_s4 → s4 (LLD) → d_s5 → s5 (PM9/10) → 
d_LP_done → LP_done (完成)
```

### Release Time Propagation

```
晶圆进入 d_s1 → 预估 s1 释放时间 → 链式传播到 s3 → 链式传播到 s5
晶圆实际进入 s1 → 更新精确释放时间 → 链式更新 s3 → 链式更新 s5
```

### Reward Calculation Flow

```
时间窗口 [t1, t2] → 遍历所有 Place → 计算各项奖励/惩罚 → 汇总总奖励
- 加工奖励：type=1 腔室在加工时间内
- 超时惩罚：超过 processing_time + Residual_time
- 预警惩罚：slack < T_warn
- 安全奖励：slack > T_safe
- 时间成本：-c_time * delta_t
```

## Constraints and Invariants

1. **容量约束**: `len(place.tokens) <= place.capacity`（始终成立）
2. **时间单调性**: `current_time >= previous_time`（时间只能向前）
3. **Token 守恒**: 变迁执行前后，总 token 数不变（消耗 = 产生）
4. **释放时间单调性**: `new_release_time >= old_release_time`（只能更新为更晚的时间）
5. **统计一致性**: `wafer_stats` 中的时间信息与实际 token 状态一致

## Performance Considerations

- **Token 队列**: 使用 `deque` 实现 O(1) 的 `popleft()` 和 `append()`
- **矩阵运算**: 使用 NumPy 向量化操作，避免 Python 循环
- **释放时间查找**: 使用 `min()` 查找最早释放时间，O(n) 但 n 通常很小（<= capacity）
- **统计计算**: 延迟计算，只在需要时遍历所有晶圆
