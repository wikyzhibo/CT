# 时间轴管理的变迁发射逻辑实现（多资源库所版本）

## 目标

修改Petri网的变迁发射逻辑，通过时间轴管理系统来：
1. 避免同一变迁在执行时间段内重叠冲突
2. 管理库所占用时间，支持多资源（机器）库所（如p3、p5容量为2，需要维护2个时间轴）
3. 允许在时间轴上插空执行变迁

## 关键需求

- **变迁时间轴**：每个变迁t有独立的时间轴列表，存储所有执行时间段
- **库所多资源时间轴**：每个库所p根据容量k[p]有多个资源（机器），每个资源有独立的时间轴
  - 例如：p3容量为2，需要维护2个资源时间轴
  - 例如：p5容量为2，需要维护2个资源时间轴
  - 容量为1的库所只有1个资源时间轴

## 实现方案

### 1. 数据结构设计

在 `__init__` 中初始化时间轴管理数据结构：

- `self.transition_times`: `List[List[Tuple[int, int]]]`，长度为 `self.T`
  - 每个元素 `transition_times[t]` 存储变迁 `t` 的所有执行时间段
  - 每个时间段表示为 `(start_time, end_time)` 元组（左闭右开区间）
  
- `self.place_times`: `List[List[List[Tuple[int, int]]]]`，长度为 `self.P`
  - 每个元素 `place_times[p]` 是一个列表，长度为 `k[p]`（库所容量）
  - `place_times[p][r]` 存储库所 `p` 的第 `r` 个资源的所有占用时间段
  - 占用时间段：从token进入库所开始，持续 `self.ptime[p]` 时间

### 2. 修改 `__init__` 方法

在 `[solutions/try/net.py](solutions/try/net.py)` 的 `__init__` 方法中（约在 `self.time = 1` 之后，第87行附近），添加：

```python
# 时间轴管理：变迁执行时间段和库所占用时间段
self.transition_times = [[] for _ in range(self.T)]  # T个变迁的时间段列表

# 库所多资源时间轴：每个库所p有k[p]个资源，每个资源有自己的时间轴
# 注意：需要在self.k初始化后才能创建，所以放在self.k初始化之后
if self.k is not None:
    self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]
else:
    # 如果k未初始化，先创建空列表，后续在_arrange_time后重新初始化
    self.place_times = []
```

**注意**：由于 `self.k` 可能在 `_arrange_time` 后才初始化，需要在 `self.k` 初始化后（约第80行后）添加：

```python
# 初始化库所多资源时间轴（在k初始化后）
if self.k is not None:
    self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]
```

### 3. 修改 `reset` 方法

在 `[solutions/try/net.py](solutions/try/net.py)` 的 `reset` 方法中（约在 `self.zb.reset()` 之后，第194行后），添加：

```python
# 重置时间轴
self.transition_times = [[] for _ in range(self.T)]
if self.k is not None:
    self.place_times = [[[] for _ in range(int(self.k[p]))] for p in range(self.P)]
```

### 4. 实现时间区间重叠检查辅助函数

在 `_earliest_enable_time` 之前（约第333行前）添加辅助函数：

```python
def _is_time_slot_available(self, start: int, end: int, occupied_intervals: List[Tuple[int, int]]) -> bool:
    """
    检查时间段 [start, end) 是否与已有时间段重叠
    
    :param start: 开始时间（含）
    :param end: 结束时间（不含）
    :param occupied_intervals: 已有时间段列表，每个元素为 (start, end)
    :return: True表示可用（无重叠），False表示不可用
    """
    for occ_start, occ_end in occupied_intervals:
        # 检查重叠：新区间的开始 < 旧区间的结束 且 新区间的结束 > 旧区间的开始
        if start < occ_end and end > occ_start:
            return False
    return True


def _find_earliest_available_time(self,
                                  earliest_start: int,
                                  duration: int,
                                  occupied_intervals: List[Tuple[int, int]]) -> int:
    """
    从 earliest_start 开始，找到最早的可用于执行 duration 时长的时刻
    
    :param earliest_start: 最早开始时间下界
    :param duration: 需要执行的时间长度
    :param occupied_intervals: 已有时间段列表（按开始时间排序）
    :return: 最早可用的开始时间
    """
    current_time = earliest_start
    end_time = current_time + duration

    # 如果没有已占用的时间段，直接返回
    if not occupied_intervals:
        return current_time

    # 尝试从 earliest_start 开始，逐步向后查找可用时间段
    max_iterations = 10000  # 防止无限循环
    for _ in range(max_iterations):
        if self._is_time_slot_available(current_time, end_time, occupied_intervals):
            return current_time

        # 找到第一个结束时间 > current_time 的区间，将 current_time 设为该区间的结束时间
        next_start = None
        for occ_start, occ_end in occupied_intervals:
            if occ_end > current_time:
                next_start = occ_end
                break

        if next_start is None:
            # 所有占用区间都在 current_time 之前，当前时间可用
            return current_time

        current_time = next_start
        end_time = current_time + duration

    return -1  # 理论上不应到达这里


def _find_earliest_available_resource_time(self,
                                           place_id: int,
                                           start_time: int,
                                           duration: int) -> Tuple[int, int]:
    """
    在库所的所有资源中，找到最早可用的资源和时间
    
    :param place_id: 库所ID
    :param start_time: 最早开始时间下界
    :param duration: 需要占用的时长
    :return: (resource_id, available_start_time) 最早可用的资源ID和开始时间
    """
    if self.k is None or place_id >= len(self.place_times):
        return (0, start_time)

    num_resources = len(self.place_times[place_id])
    best_resource = 0
    best_time = start_time
    best_available_time = self._from_interval_find_time(
        start_time, duration, self.place_times[place_id][0]
    )

    # 遍历所有资源，找到最早可用的
    for r in range(1, num_resources):
        available_time = self._from_interval_find_time(
            start_time, duration, self.place_times[place_id][r]
        )
        if available_time < best_available_time:
            best_available_time = available_time
            best_resource = r

    return (best_resource, best_available_time)
```

### 5. 修改 `_earliest_enable_time` 方法

修改 `[solutions/try/net.py](solutions/try/net.py)` 中的 `_earliest_enable_time` 方法（约第334-372行）：

**新逻辑**：
1. 计算token最早的许可时间 `t1`（基于 `enter_time + ptime`）
2. 检查变迁执行时间：查询 `transition_times[t]`，找到最早不重叠的执行时间
3. 检查库所占用时间：对于每个后置库所，在所有资源中找到最早可用的时间
4. 返回所有约束下的最早可用时间

**实现要点**：
- 对于每个前置库所，计算token可用时间：`tok_enter + self.ptime[p]`
- 取所有前置库所中的最大值作为最早开始时间下界 `t1`
- 检查变迁t在 `[t1, t1+ttime]` 是否与 `transition_times[t]` 重叠，如果重叠则找到最早可用时间
- 对于每个后置库所p，使用 `_find_earliest_available_resource_time` 找到最早可用的资源和时间
- 最终返回所有约束下的最大值

### 6. 修改 `_tpn_fire` 方法

修改 `[solutions/try/net.py](solutions/try/net.py)` 中的 `_tpn_fire` 方法（约第375-418行）：

在执行变迁后：
1. **记录变迁执行时间段**：在 `transition_times[t]` 中添加 `(te, te + d)`（执行时间段，左闭右开）
2. **记录库所占用时间段**：
   - 对于每个后置库所 `p`，当token进入时：
     - 计算token进入时间 `enter_new`
     - 计算占用时长 `duration = self.ptime[p]`
     - 使用 `_find_earliest_available_resource_time` 找到可用资源（虽然理论上在_earliest_enable_time中已经确保可用，但这里需要确定具体使用哪个资源）
     - 在对应资源的时间轴 `place_times[p][resource_id]` 中添加占用时间段 `(enter_new, enter_new + duration)`

**实现位置**：
- 在 `enter_new = tf + 1` 计算后（约第390行后）
- 在token添加到后置库所后（约第405-413行之后）

**注意**：由于在 `_earliest_enable_time` 中已经确保了时间可用，这里可以直接使用 `enter_new` 作为开始时间，但仍需要确定使用哪个资源。可以简化：对于每个后置库所，找到在 `enter_new` 时刻可用的资源（可能有多个，选择第一个可用的）。

### 7. 库所资源选择策略

在 `_tpn_fire` 中，当token进入后置库所时：
- 对于库所p，需要在 `enter_new` 时刻找到一个可用的资源
- 可以遍历所有资源，找到在 `[enter_new, enter_new + ptime[p])` 时间段内可用的资源
- 如果多个资源都可用，选择资源ID最小的（或任意一个）

可以添加辅助函数：

```python
def _find_available_resource_at_time(self, place_id: int, start_time: int, duration: int) -> int:
    """
    在指定时刻找到库所中可用的资源
    
    :param place_id: 库所ID
    :param start_time: 开始时间
    :param duration: 占用时长
    :return: 可用资源的ID，如果都不可用则返回-1（理论上不应发生）
    """
    if self.k is None or place_id >= len(self.place_times):
        return 0
    
    for r in range(len(self.place_times[place_id])):
        if self._is_time_slot_available(start_time, start_time + duration, self.place_times[place_id][r]):
            return r
    
    return -1  # 理论上不应到达这里（因为_earliest_enable_time已确保可用）
```

### 8. 注意事项

- 时间段存储为左闭右开区间：`[start, end)`，即包含 `start`，不包含 `end`
- 变迁执行时间段：`[te, te + d)`，其中 `d = self.ttime`
- 库所占用时间段：`[enter_time, enter_time + ptime[p])`
- 回溯时不需要撤销时间轴记录（用户已确认）
- 在 `reset` 时需要清空所有时间轴记录
- 对于容量为2的库所（如p3、p5），需要维护2个独立的时间轴列表

## 文件修改清单

1. `[solutions/try/net.py](solutions/try/net.py)` - `__init__` 方法：添加时间轴数据结构初始化（在self.k初始化后）
2. `[solutions/try/net.py](solutions/try/net.py)` - `reset` 方法：添加时间轴重置
3. `[solutions/try/net.py](solutions/try/net.py)` - 添加辅助函数：
   - `_is_time_slot_available`
   - `_find_earliest_available_time`
   - `_find_earliest_available_resource_time`
   - `_find_available_resource_at_time`
4. `[solutions/try/net.py](solutions/try/net.py)` - `_earliest_enable_time` 方法：实现基于时间轴的查询逻辑
5. `[solutions/try/net.py](solutions/try/net.py)` - `_tpn_fire` 方法：添加时间段记录逻辑

