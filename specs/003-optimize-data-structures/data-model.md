# Data Model: 在 002-sim-speedup 基础上优化数据结构

**Date**: 2026-01-26  
**Feature**: 003-optimize-data-structures

## Overview

本文档描述了优化后的数据结构模型，包括 `Place`、`BasedToken` 和相关的缓存结构。优化主要关注内存布局、访问模式和缓存策略。

## Entities

### Place (优化后)

**Purpose**: 表示 Petri 网中的一个库所（Place），存储 token 队列和释放时间调度信息。

**Fields**:
- `name: str` - 库所名称（不可变）
- `capacity: int` - 库所容量（不可变）
- `processing_time: int` - 加工时间（不可变）
- `type: int` - 库所类型：1=加工腔室, 2=运输库所, 3=空闲库所, 4=资源库所, 5=无驻留约束腔室（不可变）
- `tokens: Deque[BasedToken]` - Token 队列（可变）
- `release_schedule: Deque[Tuple[int, int]]` - 释放时间调度队列：[(token_id, release_time), ...]（可变）
- `last_machine: int` - 上次分配的机器编号（仅 type=1 使用），用于轮换分配机器（可变）

**Optimizations**:
- 使用 `__slots__` 减少内存占用和属性访问开销
- `tokens` 和 `release_schedule` 保持为 `deque`（不能放入 `__slots__`）

**Validation Rules**:
- `capacity > 0`
- `processing_time >= 0`
- `type` 必须在 [1, 2, 3, 4, 5] 范围内
- `len(tokens) <= capacity`
- `last_machine` 在 [-1, capacity-1] 范围内（type=1 时）

**Relationships**:
- 一个 `Place` 包含多个 `BasedToken`（通过 `tokens` 队列）
- 一个 `Place` 属于一个 `Petri` 实例（通过 `marks` 列表）

### BasedToken (优化后)

**Purpose**: 表示 Petri 网中的一个 token，包含进入时间、停留时间、token ID 和机器编号。

**Fields**:
- `enter_time: int` - 进入时间（可变）
- `stay_time: int` - 停留时间（可变）
- `token_id: int` - Token 唯一标识，-1 表示未分配（可变）
- `machine: int` - 分配的机器编号，-1 表示未分配（可变）

**Optimizations**:
- 使用 `__slots__` 减少内存占用和属性访问开销

**Validation Rules**:
- `enter_time >= 0`
- `stay_time >= 0`
- `token_id >= -1`
- `machine >= -1`

**Relationships**:
- 一个 `BasedToken` 属于一个 `Place`（通过 `tokens` 队列）

### MarksByTypeCache

**Purpose**: 按类型分组缓存 `marks` 列表，避免频繁遍历所有库所。

**Fields**:
- `_marks_by_type: Dict[int, List[Place]]` - 按类型分组的库所列表缓存

**Structure**:
```python
{
    1: [Place(...), Place(...)],  # type=1 的加工腔室
    2: [Place(...)],              # type=2 的运输库所
    3: [Place(...)],              # type=3 的空闲库所
    4: [Place(...)],              # type=4 的资源库所
    5: [Place(...), Place(...)], # type=5 的无驻留约束腔室
}
```

**Optimizations**:
- 在 `__init__` 中构建缓存
- 在 `reset` 中更新缓存（如果 `marks` 列表发生变化）
- 在奖励计算、状态更新等操作中使用缓存，避免遍历所有库所

**Validation Rules**:
- 缓存中的 `Place` 对象必须与 `marks` 列表中的对象一致
- 缓存必须包含所有类型的库所（即使列表为空）

**Relationships**:
- 缓存引用 `marks` 列表中的 `Place` 对象（不复制）

### PrePstPlacesCache (已存在，优化访问)

**Purpose**: 缓存变迁的前置和后置库所索引，避免重复计算。

**Fields**:
- `_pre_places_cache: Dict[int, np.ndarray]` - 前置库所索引缓存
- `_pst_places_cache: Dict[int, np.ndarray]` - 后置库所索引缓存

**Optimizations**:
- 使用局部变量缓存字典引用，减少属性查找开销
- 使用 `np.flatnonzero` 而非 `np.nonzero` 减少维度

**Validation Rules**:
- 缓存的索引必须在有效范围内 [0, P-1]，其中 P 是库所数量

**Relationships**:
- 缓存引用 `pre` 和 `pst` 矩阵中的索引

## State Transitions

### Place State

**Initial State**: 
- `tokens` 为空或包含初始 token
- `release_schedule` 为空
- `last_machine = -1`

**Token 进入**:
- `tokens.append(token)`
- 如果是有驻留约束的库所，可能更新 `release_schedule`

**Token 离开**:
- `tokens.popleft()` 或 `tokens.pop()`
- 从 `release_schedule` 中移除对应记录

**Reset**:
- `tokens` 重置为初始状态
- `release_schedule` 清空
- `last_machine = -1`

### BasedToken State

**Initial State**:
- `enter_time = 0` 或指定值
- `stay_time = 0`
- `token_id = -1` 或指定值
- `machine = -1` 或指定值

**时间推进**:
- `stay_time` 更新为 `current_time - enter_time`

**Reset**:
- 所有字段重置为初始值

## Data Access Patterns

### 频繁访问模式

1. **按类型访问库所**:
   - 使用 `_marks_by_type[type]` 而非遍历 `marks`
   - 适用于奖励计算、状态更新等操作

2. **通过索引访问库所**:
   - 使用 `marks[p_idx]` 直接访问
   - 适用于使能检查、变迁执行等操作

3. **访问 token 属性**:
   - 使用 `token.enter_time` 等属性访问
   - `__slots__` 优化后访问更快

4. **访问字典映射**:
   - 使用局部变量缓存字典引用
   - 适用于 `id2p_name`、`id2t_name` 等

### 缓存策略

1. **按类型分组缓存**:
   - 构建时间：O(n)，其中 n 是库所数量
   - 访问时间：O(1) 到 O(k)，其中 k 是特定类型的库所数量
   - 更新频率：仅在 `reset` 时更新

2. **前置/后置库所缓存**:
   - 构建时间：O(1)（延迟构建）
   - 访问时间：O(1)（字典查找）
   - 更新频率：不需要更新（网络结构不变）

## Memory Considerations

### 内存优化

1. **`__slots__` 优化**:
   - `Place` 对象内存占用减少约 30-50%
   - `BasedToken` 对象内存占用减少约 30-50%

2. **缓存内存开销**:
   - `_marks_by_type`: O(n) 额外内存，其中 n 是库所数量
   - `_pre_places_cache`: O(T) 额外内存，其中 T 是变迁数量
   - `_pst_places_cache`: O(T) 额外内存

3. **总体内存使用**:
   - 目标：内存使用增加不超过 10%
   - 通过 `__slots__` 节省的内存可以部分抵消缓存的内存开销

## Compatibility

### 与 002-sim-speedup 的兼容性

- 所有优化措施与 002-sim-speedup 的优化措施兼容
- 可以同时启用两种优化措施
- 缓存结构不会冲突（使用不同的缓存键）

### 向后兼容性

- 所有公共 API 保持不变
- 内部实现可以优化，但不改变外部接口
- 序列化/反序列化保持兼容（如果使用）
